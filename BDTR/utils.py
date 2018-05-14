import tensorflow as tf
import sys

sys.path.append('/home/comp/mangye/local/lib64/python2.7/site-packages')
from sklearn.metrics import average_precision_score
import os
from collections import defaultdict
import numpy as np
import cv2


# fine_tune two modalities with one single alexnet
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
    # output_imgs =  np.ndarray([60, crop_size, crop_size, 3])   
    # for i in xrange(60):
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
    
def compute_accuracy(dist, query_ids, gallery_ids,topk):

    single_gallery_shot=False
    separate_camera_set=False
    first_match_break=True
    # distmat = dist
    distmat = dist.eval()
    m, n = distmat.shape
    # Fill up default values
    query_cams = np.zeros(m).astype(np.int32)
    gallery_cams = np.ones(n).astype(np.int32)
    # Ensure numpy array
    query_ids = np.asarray(query_ids)
    gallery_ids = np.asarray(gallery_ids)
    query_cams = np.asarray(query_cams)
    gallery_cams = np.asarray(gallery_cams)
    # Sort and find correct matches
    indices = np.argsort(distmat, axis=1)
    matches = (gallery_ids[indices] == query_ids[:, np.newaxis])
    # # Compute AP for each query

    # topk = 100    
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
        
        # # Compute CMC
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
    
    # print('top-1: {:.2%} | top-5: {:.2%} | top-10: {:.2%}| top-20: {:.2%}'.format(cmc[0], cmc[4], cmc[9], cmc[19]))
    # print('mAP: {:.2%}'.format(mAP))
    return cmc, mAP