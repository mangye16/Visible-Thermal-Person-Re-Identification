import tensorflow as tf
import sys
from model2 import Model
from network import *
from datetime import datetime
import os
from sklearn.metrics import average_precision_score

import numpy as np
import cv2
import h5py
from collections import defaultdict

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def get_test_data(input_data_path):
  
    crop_size = 227
    # self.mean = np.array([104., 117., 124.]) # original
    mean_img = np.array([123.68, 116.779, 103.939]) # ours
    with open(input_data_path) as f:
        data_file_list = open(input_data_path, 'rt').read().splitlines()
        # Get full list of color image and labels
        data_file_image = [s.split(' ')[0] for s in data_file_list]
        data_file_label = [int(s.split(' ')[1]) for s in data_file_list]
    
    output_imgs =  np.ndarray([len(data_file_image), crop_size, crop_size, 3])   
    for i in xrange(len(data_file_image)):
        # read imgs
        img = cv2.imread('../Dataset/' + data_file_image[i])
        h, w, c = img.shape
        assert c==3
        
        img = cv2.resize(img, (crop_size, crop_size))
        img = img.astype(np.float32)
        img -= mean_img
        output_imgs[i] = img
    # data_file_label = data_file_label[:60]
    return   output_imgs,  data_file_label
    
def _unique_sample(ids_dict, num):
    mask = np.zeros(num, dtype=np.bool)
    for _, indices in ids_dict.items():
        i = np.random.choice(indices)
        mask[i] = True
    return mask

def compute_accuracy(dist, query_ids, gallery_ids, topk = 20):

    single_gallery_shot=False
    first_match_break=True
    separate_camera_set=False
    distmat = dist.eval()
    m, n = distmat.shape
    # Fill up default values
    query_cams = np.zeros(m).astype(np.int32)
    gallery_cams = 2 * np.ones(n).astype(np.int32)
    # Ensure numpy array
    query_ids = np.asarray(query_ids)
    gallery_ids = np.asarray(gallery_ids)
    query_cams = np.asarray(query_cams)
    gallery_cams = np.asarray(gallery_cams)
    # Sort and find correct matches
    indices = np.argsort(distmat, axis=1)
    matches = (gallery_ids[indices] == query_ids[:, np.newaxis])
    
    # Compute AP for each query   
    ret = np.zeros(topk)
    num_valid_queries = 0
    aps = []
    for i in range(m):
        # Filter out the same id and same camera
        valid = ((gallery_ids[indices[i]] != query_ids[i]) |(gallery_cams[indices[i]] != query_cams[i]))
        if not np.any(matches[i, valid]): continue
        # Compute mAP
        y_true = matches[i, valid]
        y_score = -distmat[i][indices[i]][valid]
        aps.append(average_precision_score(y_true, y_score))
        
        # Compute CMC  
        if separate_camera_set:
            # Filter out samples from same camera
            valid &= (gallery_cams[indices[i]] != query_cams[i])
            
        if single_gallery_shot:
            repeat = 10
            gids = gallery_ids[indices[i][valid]]
            inds = np.where(valid)[0]
            ids_dict = defaultdict(list)
            for j, x in zip(inds, gids):
                ids_dict[x].append(j)
        else:
            repeat = 1
        for _ in range(repeat):
            if single_gallery_shot:
                # Randomly choose one instance for each id
                sampled = (valid & _unique_sample(ids_dict, len(valid)))
                index = np.nonzero(matches[i, sampled])[0]
            else:
                index = np.nonzero(matches[i, valid])[0]
            delta = 1. / (len(index) * repeat)
            for j, k in enumerate(index):
                if k - j >= topk: break
                if first_match_break:
                    ret[k - j] += 1
                    break
                ret[k - j] += delta
        num_valid_queries += 1

    mAP = np.mean(aps)
    cmc = ret.cumsum() / num_valid_queries
    return cmc, mAP


def main():  
    # Dataset path
    test_thermal_list = '../idx/test_thermal_1.txt'
    test_color_list = '../idx/test_color_1.txt'
    dataset_path = '../Dataset/'
    model_path = 'log/tone_iter1_900.ckpt'
    save_dir = 'data/'

    test_num = 2060
    # Graph input
    x1 = tf.placeholder(tf.float32, [test_num, 227, 227, 3])
    x2 = tf.placeholder(tf.float32, [test_num, 227, 227, 3])    
    keep_var = tf.placeholder(tf.float32)   
    
    # Model
    predict1 = Model().alexnet_visible(x1, keep_var)
    predict2 = Model().alexnet_thermal(x2, keep_var)
    feat     = Model().share_modal(predict1, predict2, keep_var)
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
        feature1 = sess.run(predict1, feed_dict={ x1:test_color_imgs, keep_var:  1. })          
        feature2 = sess.run(predict2, feed_dict={ x2:test_thermal_imgs, keep_var:  1.})
        feature  = sess.run(feat, feed_dict={ predict1:feature1, predict2:feature2, keep_var:  1.})       
        feature1, feature2 = tf.split(0, 2, feature)
        
        print 'Evaluate Performance'
        query_t_norm = tf.nn.l2_normalize(feature1, dim=1)
        test_t_norm  = tf.nn.l2_normalize(feature2, dim=1)

        distmat = tf.matmul(query_t_norm, test_t_norm, transpose_a=False, transpose_b=True)

        cmc, mAP = compute_accuracy(-distmat, test_color_labels[:test_num], test_thermal_labels[:test_num],topk = 20)
            
        print('top-1: {:.2%} | top-5: {:.2%} | top-10: {:.2%}| top-20: {:.2%}'.format(cmc[0], cmc[4], cmc[9], cmc[19]))
        print('mAP: {:.2%}'.format(mAP))
        
        # # save feature
        print 'Save Feature'
        feature = query_t_norm.eval()
        f = h5py.File(save_dir + 'train_color_iter_1.mat','w')
        f.create_dataset('feature',data=feature)
        f.close()
        
        feature = test_t_norm.eval()
        f = h5py.File(save_dir + 'train_thermal_iter_1.mat','w')
        f.create_dataset('feature',data=feature)
        f.close()
 
if __name__ == '__main__':
    main() 
    




