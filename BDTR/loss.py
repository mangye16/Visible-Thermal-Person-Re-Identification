

import tensorflow as tf
import sys

sys.path.append('/home/comp/mangye/local/lib64/python2.7/site-packages')
from sklearn.metrics import average_precision_score
import os
from collections import defaultdict
import numpy as np
import cv2


def all_diffs(a, b):
    return tf.expand_dims(a, 1) - tf.expand_dims(b, 0)


def cdist(a, b, metric='euclidean'):

    with tf.name_scope("cdist"):
        diffs = all_diffs(a, b)
        if metric == 'sqeuclidean':
            return tf.reduce_sum(tf.square(diffs), axis=-1)
        elif metric == 'euclidean':
            return tf.sqrt(tf.reduce_sum(tf.square(diffs), -1) + 1e-12)
        elif metric == 'cosine':
            return 1 - tf.matmul(a, b, transpose_a=False, transpose_b=True)
        elif metric == 'cityblock':
            return tf.reduce_sum(tf.abs(diffs), axis=-1)
        else:
            raise NotImplementedError(
                'The following metric is not implemented by `cdist` yet' )
                
def gather_2d(params, indices):
    # only for two dim now
    shape = params.get_shape().as_list()
    assert len(shape) == 2, 'only support 2d matrix'
    if shape[1]>0:
        flat = tf.reshape(params, [np.prod(shape)])
        flat_idx = tf.slice(indices, [0,0], [shape[0],1]) * shape[1] + tf.slice(indices, [0,1], [shape[0],1])
        flat_idx = tf.reshape(flat_idx, [flat_idx.get_shape().as_list()[0]])
        return tf.gather(flat, flat_idx)
    else:
        return 0
                               
def compute_birank_loss(feat1, feat2, batch_size, margin = 0.5, lambda_intra = 0.1):

    y1 = [x for x in range(batch_size)]
    y2 = [x for x in range(batch_size)]
    
    feat1 = tf.nn.l2_normalize(feat1, dim=1)
    feat2 = tf.nn.l2_normalize(feat2, dim=1)
    
    # computer the positive and negative mask
    same_identity_mask = tf.equal(tf.expand_dims(y1, 1),tf.expand_dims(y2, 0))
    negative_mask = tf.logical_not(same_identity_mask)
    positive_mask = same_identity_mask
        
    # from 1 to 2
    dists_vt = cdist(feat1, feat2)
    furthest_positive = tf.reduce_max(dists_vt*tf.cast(positive_mask, tf.float32), 1)
    pos_positive = tf.argmax(dists_vt*tf.cast(positive_mask, tf.float32), 1)
    # closest_negative = tf.map_fn(lambda x: tf.reduce_min(tf.boolean_mask(x[0], x[1])),(dists_vt, negative_mask), tf.float32)
    closest_negative = tf.reduce_min(dists_vt + 1e5*tf.cast(same_identity_mask, tf.float32), 1)
    pos_negative = tf.argmin(dists_vt + 1e5*tf.cast(same_identity_mask, tf.float32), 1)
    
    # pdb.set_trace()
     # computer cross modality loss
    cross_diff1 = furthest_positive - closest_negative
    cross_diff1 = tf.maximum(cross_diff1 + margin, 0.0)
    loss_mean1 = tf.reduce_mean(cross_diff1)
    
    # intra-modality constraints
    intra_dist1 = cdist(feat2, feat2)
    idx1 = tf.concat(1,[tf.reshape(pos_positive,[batch_size,1]),tf.reshape(pos_negative,[batch_size,1])])
   
    
    intra_diff1 = gather_2d(intra_dist1,idx1)
    intra_diff1 = tf.maximum(0.1 - intra_diff1, 0.0)
    intra_loss1 = tf.reduce_mean(intra_diff1)
    
    # compute precision at top-1
    diff_mat = tf.less(furthest_positive ,closest_negative)
    prec1 = tf.reduce_mean(tf.cast(diff_mat, tf.float32))
    
    
    # from 2 to 1
    dists_tv = cdist(feat2, feat1)
    furthest_positive = tf.reduce_max(dists_tv*tf.cast(positive_mask, tf.float32), 1)
    pos_positive = tf.argmax(dists_tv*tf.cast(positive_mask, tf.float32), 1)
    # closest_negative = tf.map_fn(lambda x: tf.reduce_min(tf.boolean_mask(x[0], x[1])),(dists_tv, negative_mask), tf.float32)
    closest_negative = tf.reduce_min(dists_tv + 1e5*tf.cast(same_identity_mask, tf.float32), 1)
    pos_negative = tf.argmin(dists_tv + 1e5*tf.cast(same_identity_mask, tf.float32), 1)
    
    # computer cross modality loss
    cross_diff2 = furthest_positive - closest_negative
    cross_diff2 = tf.maximum(cross_diff2 + margin, 0.0)
    loss_mean2 = tf.reduce_mean(cross_diff2)
    
    # intra-modality constraints
    intra_dist2 = cdist(feat1, feat1)
    idx2 = tf.concat(1,[tf.reshape(pos_positive,[batch_size,1]),tf.reshape(pos_negative,[batch_size,1])])
   
    intra_diff2 = gather_2d(intra_dist2,idx2)
    intra_diff2 = tf.maximum(0.1 - intra_diff2, 0.0)
    intra_loss2 = tf.reduce_mean(intra_diff2)
    
    # compute precision at top-1
    diff_mat = tf.less(furthest_positive ,closest_negative)
    prec2 = tf.reduce_mean(tf.cast(diff_mat, tf.float32))
    

    prec = tf.add(prec1, prec2)/2.
    inter_loss = tf.add(loss_mean1, loss_mean2)   
    intra_loss = tf.add(intra_loss1, intra_loss2)
    
    loss_sum = tf.add(inter_loss, lambda_intra*intra_loss)  
    
    return loss_sum, prec

