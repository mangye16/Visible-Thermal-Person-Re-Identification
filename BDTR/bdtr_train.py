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

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def main():
    # Dataset path
	# Visible Images and Color Images
    train_color_list   = '../idx/train_color_2tream_1.txt'
    train_thermal_list = '../idx/train_thermal_2tream_1.txt'

    checkpoint_path = 'log/'
    # Learning params
    learning_rate = 0.001
    training_iters = 5000
    batch_size = 64
    display_step = 5
    test_step = 300 # 0.5 epoch
    
    # Network params
    n_classes = 206
    keep_rate = 0.5

    # Graph input
    x1 = tf.placeholder(tf.float32, [batch_size, 227, 227, 3])
    x2 = tf.placeholder(tf.float32, [batch_size, 227, 227, 3])
    y0 = tf.placeholder(tf.float32, [batch_size*2, n_classes])
    # y  = tf.placeholder(tf.float32, [batch_size, 1])
    y1 = tf.placeholder(tf.float32, [batch_size, 1])
    y2 = tf.placeholder(tf.float32, [batch_size, 1])
    
    configProt = tf.ConfigProto()
    sess = tf.Session(config = configProt)
    
    keep_var = tf.placeholder(tf.float32)

    # Model
    feat1 = Model().alexnet_visible(x1, keep_var)    
    feat2 = Model().alexnet_thermal(x2, keep_var)
    feat, pred0 = Model().modal_embedding(feat1, feat2, keep_var)
    
    # norm_feat = tf.nn.l2_normalize(feat, 1, epsilon=1e-12)
    feature1, feature2 = tf.split(0, 2, feat)
    rank_loss, prec = compute_birank_loss(feature1, feature2, batch_size)
    
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
        
    # Launch the graph
    with tf.Session() as sess:
        print 'Init variable'
        
        sess.run(init)
        print 'Start training'
        step = 0
        while step < training_iters:
            batch_x1, batch_x2, bacth_y0, batch_y1, batch_y2 = dataset.next_batch(batch_size, 'train')

            sess.run(optimizer, feed_dict={x1: batch_x1, x2: batch_x2, y0:bacth_y0, y1: batch_y1, y2: batch_y2, keep_var: keep_rate})
           
            # Display training status
            if step%display_step == 0:
                acc = sess.run(prec, feed_dict={x1: batch_x1, x2: batch_x2, y0:bacth_y0, y1: batch_y1, y2: batch_y2, keep_var: 1.0})
                batch_loss = sess.run(total_loss, feed_dict={x1: batch_x1, x2: batch_x2, y0:bacth_y0, y1: batch_y1, y2: batch_y2, keep_var: 1.0})

                print >> sys.stderr, "{} Iter {}: Training Loss = {:.4f}, Top-1 Accuracy = {:.4f}".format(datetime.now(), step, batch_loss, acc)
                # print >> sys.stderr, "diff: {} ".format(diff)
                
            if step % 100 == 0: 
                 # for iround in range                
                print 'A short Test'
                feature1 = sess.run(feat1, feed_dict={x1:test_color_imgs[:batch_size],   keep_var: 1.}) 
                feature2 = sess.run(feat2, feed_dict={x2:test_thermal_imgs[:batch_size], keep_var: 1.})

                feature  = sess.run(feat, feed_dict={feat1:feature1, feat2:feature2, keep_var: 1.})
                
                test_color_feature, test_thermal_feature = tf.split(0, 2, feature)
                # test_feat = sess.run(feat, feed_dict={feat1: test_color_feature, feat2:test_thermal_feature  keep_var: 1.})
                query_t_norm = tf.nn.l2_normalize(test_color_feature, dim=1)
                test_t_norm  = tf.nn.l2_normalize(test_thermal_feature, dim=1)
                distmat = tf.matmul(query_t_norm, test_t_norm, transpose_a=False, transpose_b=True)
                
                cmc, mAP = compute_accuracy(-distmat, test_color_labels[:batch_size], test_thermal_labels[:batch_size],topk=20)
                
                print('top-1: {:.2%} | top-5: {:.2%} | top-10: {:.2%}| top-20: {:.2%}'.format(cmc[0], cmc[4], cmc[9], cmc[19]))
                print('mAP: {:.2%}'.format(mAP))
            
            # Save the model checkpoint periodically.
            if step % 2500 == 0 or step == training_iters:
                checkpoint_name = os.path.join(checkpoint_path, 'cam_bdtr_trial_1_'+ str(step) +'.ckpt')
                save_path = saver.save(sess, checkpoint_name)

            step += 1

        print "Finish!"

if __name__ == '__main__':
    main()

