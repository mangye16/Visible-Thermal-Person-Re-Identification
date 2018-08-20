import tensorflow as tf
import sys
from model import Model
from dataset2 import Dataset
from network import *
from datetime import datetime
from utils import *
from loss import *
sys.path.append('/home/comp/mangye/local/lib64/python2.7/site-packages')
from sklearn.metrics import average_precision_score
import os
from collections import defaultdict
import numbers
import numpy as np
import cv2
import pdb

os.environ["CUDA_VISIBLE_DEVICES"] = "2"


def main():
    # Dataset path
	# Visible Images and Color Images
    trial = 1
    train_color_list   = '../idx/train_color_{}'.format(trial)+ '.txt'
    train_thermal_list = '../idx/train_thermal_{}'.format(trial)+ '.txt'

    checkpoint_path = 'save_model/'
    # Learning params
    learning_rate = 0.001
    training_iters = 5000
    batch_size = 64
    display_step = 5
    test_step = 500 # 0.5 epoch
    margin = 0.5
    lambda_intra = 0.1
    feature_dim = 1024
    test_batch_size = batch_size
    
    # Network params
    n_classes = 206
    keep_rate = 0.5
    
    suffix = 'BDTR'
    # model params
    suffix = suffix + '_drop_{}'.format(keep_rate) 
    suffix = suffix + '_lr_{:1.1e}'.format(learning_rate) 
    suffix = suffix + '_margin_{}'.format(margin)
    suffix = suffix + '_batch_{}'.format(batch_size)
    suffix = suffix + '_w_intra_{}'.format(lambda_intra)
    suffix = suffix + '_trial_{}_'.format(trial)
    

    # Graph input
    x1 = tf.placeholder(tf.float32, [None, 227, 227, 3])
    x2 = tf.placeholder(tf.float32, [None, 227, 227, 3])
    y0 = tf.placeholder(tf.float32, [None, n_classes])
    # y  = tf.placeholder(tf.float32, [batch_size, 1])
    y1 = tf.placeholder(tf.float32, [None, 1])
    y2 = tf.placeholder(tf.float32, [None, 1])
    
    configProt = tf.ConfigProto()
    sess = tf.Session(config = configProt)
    
    keep_var = tf.placeholder(tf.float32)

    # Construct Model
    feat1 = Model().alexnet_visible(x1, keep_var)    
    feat2 = Model().alexnet_thermal(x2, keep_var)
    feat, pred0 = Model().modal_embedding(feat1, feat2, keep_var, feature_dim)
    
    # norm_feat = tf.nn.l2_normalize(feat, 1, epsilon=1e-12)
    feature1, feature2 = tf.split(0, 2, feat)
    # pdb.set_trace()
    rank_loss, prec = compute_birank_loss(feature1, feature2, batch_size, margin, lambda_intra)
    
    # Loss and optimizer
    identity_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred0, y0))
    
	# #################################################################################
	# Total Loss: 
	# 	Ranking loss performs much better on small dataset
	# 	Identity loss is better for large scale dataset with abundant training samples
	# 	We could adjust the combining weights to achieve different performance 
	# #################################################################################
	
    total_loss = identity_loss + rank_loss
    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9).minimize(total_loss)

    # Evaluation
    correct_pred0 = tf.equal(tf.argmax(pred0, 1), tf.argmax(y0, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred0, tf.float32))

    # Init    
    init = tf.initialize_all_variables()
    saver = tf.train.Saver()
    
    # Load dataset
    dataset = Dataset(train_color_list, train_thermal_list)
    
    # load testing data (A short test for fast visualization)
    test_thermal_list = '../idx/test_thermal_1.txt'
    test_color_list = '../idx/test_color_1.txt'
    
    test_color_imgs,   test_color_labels   = get_test_data(test_color_list)
    test_thermal_imgs, test_thermal_labels = get_test_data(test_thermal_list)
    

    if not os.path.isdir(checkpoint_path):
        os.makedirs(checkpoint_path)
       

    # Launch the graph
    with tf.Session() as sess:
        print 'Init variable'
        
        sess.run(init)
        print 'Start training'
        step = 1
        while step < training_iters:
            batch_x1, batch_x2, bacth_y0, batch_y1, batch_y2 = dataset.next_batch(batch_size, 'train')

            sess.run(optimizer, feed_dict={x1: batch_x1, x2: batch_x2, y0:bacth_y0, y1: batch_y1, y2: batch_y2, keep_var: keep_rate})
           
            # Display training status
            if step%display_step == 0:
                acc = sess.run(prec, feed_dict={x1: batch_x1, x2: batch_x2, y0:bacth_y0, y1: batch_y1, y2: batch_y2, keep_var: 1.0})
                batch_loss = sess.run(total_loss, feed_dict={x1: batch_x1, x2: batch_x2, y0:bacth_y0, y1: batch_y1, y2: batch_y2, keep_var: 1.0})

                print >> sys.stderr, "{} Iter {}: Training Loss = {:.4f}, Top-1 Accuracy = {:.4f}".format(datetime.now(), step, batch_loss, acc)
                # print >> sys.stderr, "diff: {} ".format(diff)
                
            if step % test_step == 0: 
                 # for iround in range                
                print ('Test Step: {}'.format(step))
                # get test color feature
                num_color_sample = test_color_imgs.shape[0]
                ptr1 = 0
                test_color_feat = np.zeros((num_color_sample,feature_dim))
                # logits = np.zeros((num_color_sample,num_category))
                print ('Extracting test color features...')
                for num in range(int(num_color_sample / test_batch_size)+1):
                    real_size = min(test_batch_size, num_color_sample-ptr1)
                    batch_test_color_imgs = test_color_imgs[ptr1:ptr1+real_size,:,:,:]
                    feature_color = sess.run(feat1, feed_dict={x1:batch_test_color_imgs,   keep_var: 1.})
                    feature  = sess.run(feat, feed_dict={feat1:feature_color, feat2:feature_color, keep_var: 1.})
                    batch_color_feat, _ = tf.split(0, 2, feature)
                    test_color_feat[ptr1:ptr1 + real_size, :] = batch_color_feat.eval()
                    ptr1 += real_size

                # get test thermal feature
                num_thermal_sample = test_thermal_imgs.shape[0]
                ptr2 = 0
                test_thermal_feat = np.zeros((num_thermal_sample,feature_dim))
                # logits = np.zeros((num_color_sample,num_category))
                print ('Extracting test thermal features...')
                for num in range(int(num_thermal_sample / test_batch_size)+1):
                    real_size = min(test_batch_size, num_thermal_sample-ptr2)
                    batch_test_thermal_imgs = test_thermal_imgs[ptr2:ptr2+real_size,:,:,:]
                    feature_thermal = sess.run(feat2, feed_dict={x2:batch_test_thermal_imgs,   keep_var: 1.})
                    feature  = sess.run(feat, feed_dict={feat1:feature_thermal, feat2:feature_thermal, keep_var: 1.})
                    _, batch_thermal_feat = tf.split(0, 2, feature)
                    test_thermal_feat[ptr2:ptr2 + real_size, :] = batch_thermal_feat.eval()
                    ptr2 += real_size
                    
                query_t_norm = tf.nn.l2_normalize(test_color_feat, dim=1)
                test_t_norm  = tf.nn.l2_normalize(test_thermal_feat, dim=1)
                distmat = tf.matmul(query_t_norm, test_t_norm, transpose_a=False, transpose_b=True)
                
                cmc, mAP = compute_accuracy(-distmat, test_color_labels, test_thermal_labels,topk=20)
                
                print('top-1: {:.2%} | top-5: {:.2%} | top-10: {:.2%}| top-20: {:.2%}'.format(cmc[0], cmc[4], cmc[9], cmc[19]))
                print('mAP: {:.2%}'.format(mAP))
            
            # Save the model checkpoint periodically.
            if step % 2500 == 0 or step == training_iters:
                checkpoint_name = os.path.join(checkpoint_path, suffix + str(step) +'.ckpt')
                save_path = saver.save(sess, checkpoint_name, write_meta_graph=False)

            step += 1

        print "Finish!"

if __name__ == '__main__':
    main()



