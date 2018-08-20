import tensorflow as tf
import sys
from model import Model
from network import *
from datetime import datetime
import os
sys.path.append('/home/comp/mangye/local/lib64/python2.7/site-packages')
from sklearn.metrics import average_precision_score
import numpy as np
import cv2
import h5py
from collections import defaultdict
from utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def main():  
    # Dataset path
    test_thermal_list = '../idx/test_thermal_1.txt'
    test_color_list   = '../idx/test_color_1.txt'
    dataset_path = '../Dataset/'
    model_path = 'log/cam_bdtr_trial_1_2500.ckpt'
    

    feature_dim = 1024
    test_batch_size = 64
    # Graph input
    x1 = tf.placeholder(tf.float32, [None, 227, 227, 3])
    x2 = tf.placeholder(tf.float32, [None, 227, 227, 3])    
    keep_var = tf.placeholder(tf.float32)   
    
    # Model
    feat1 = Model().alexnet_visible(x1, keep_var)
    feat2 = Model().alexnet_thermal(x2, keep_var)
    feat, _ = Model().modal_embedding(feat1, feat2, keep_var)
    # feat     = Model().modal_embedding(predict1, predict2, keep_var)
    # 
 
    # load model
    init = tf.initialize_all_variables()
    saver = tf.train.Saver()
       
    with tf.Session() as sess:
        sess.run(init)
        saver.restore(sess, model_path)

        print 'Load Testing Data'
        test_color_imgs,   test_color_labels   = get_test_data(test_color_list)
        test_thermal_imgs, test_thermal_labels = get_test_data(test_thermal_list)

        print 'Extracting Feature'
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

        # distmat = tf.matmul(test_t_norm, query_t_norm, transpose_a=False, transpose_b=True)
        distmat = tf.matmul(query_t_norm, test_t_norm, transpose_a=False, transpose_b=True)

        cmc, mAP = compute_accuracy(-distmat, test_color_labels, test_thermal_labels,topk = 20)
            
        print('top-1: {:.2%} | top-5: {:.2%} | top-10: {:.2%}| top-20: {:.2%}'.format(cmc[0], cmc[4], cmc[9], cmc[19]))
        print('mAP: {:.2%}'.format(mAP))
 
if __name__ == '__main__':
    main() 
    
