import tensorflow as tf
import numpy as np
import sys
from network import *
import pdb

class Model:
    @staticmethod 
    def alexnet_visible(X1, _dropout, feature=True):
        # TODO weight decay loss tern
        conv1_1 = conv(X1, 11, 11, 96, 4, 4, padding='VALID', name='conv1_1')
            
        pool1_1 = max_pool(conv1_1, 3, 3, 2, 2, padding='VALID', name='pool1_1')
        norm1_1 = lrn(pool1_1, 2, 2e-05, 0.75, name='norm1_1')
        
        # Layer 2 (conv-relu-pool-lrn)
        # conv2 = conv(norm1, 5, 5, 256, 1, 1, group=2, name='conv2')
        conv2_1 = conv(norm1_1, 5, 5, 256, 1, 1, group=2, name='conv2_1')
            
        pool2_1 = max_pool(conv2_1, 3, 3, 2, 2, padding='VALID', name='pool2_1')
        norm2_1 = lrn(pool2_1, 2, 2e-05, 0.75, name='norm2_1')
        
        # Layer 3 (conv-relu)
        # conv3 = conv(norm2, 3, 3, 384, 1, 1, name='conv3')
        conv3_1 = conv(norm2_1, 3, 3, 384, 1, 1, name='conv3_1')
        
        # Layer 4 (conv-relu)
        # conv4 = conv(conv3, 3, 3, 384, 1, 1, group=2, name='conv4')
        conv4_1 = conv(conv3_1, 3, 3, 384, 1, 1, group=2, name='conv4_1')
        
        # Layer 5 (conv-relu-pool)
        # conv5 = conv(conv4, 3, 3, 256, 1, 1, group=2, name='conv5')
        conv5_1 = conv(conv4_1, 3, 3, 256, 1, 1, group=2, name='conv5_1')

        
        pool5_1 = max_pool(conv5_1, 3, 3, 2, 2, padding='VALID', name='pool5_1')
        # Layer 6 (fc-relu-drop)
        fc6_1 = tf.reshape(pool5_1, [-1, 6*6*256])
        
        fc6r_1 = fc(fc6_1, 6*6*256, 4096, name='fc6_1')
        # fc6r_1 = tf.nn.relu(fc6r_1)
        fc6r_1 = dropout(fc6r_1, _dropout)

        return fc6r_1

    @staticmethod  
    def alexnet_thermal(X2, _dropout, feature=True):
        # TODO weight decay loss tern
        
        # Layer 1 (conv-relu-pool-lrn)
        # conv1_1 = conv(X1, 11, 11, 96, 4, 4, padding='VALID', name='conv1_1')
        conv1_2 = conv(X2, 11, 11, 96, 4, 4, padding='VALID', name='conv1_2')
            
        pool1_2 = max_pool(conv1_2, 3, 3, 2, 2, padding='VALID', name='pool1_2')
        norm1_2 = lrn(pool1_2, 2, 2e-05, 0.75, name='norm1_2')
        
        # Layer 2 (conv-relu-pool-lrn)
        # conv2 = conv(norm1, 5, 5, 256, 1, 1, group=2, name='conv2')
        conv2_2 = conv(norm1_2, 5, 5, 256, 1, 1, group=2, name='conv2_2')
            
        pool2_2 = max_pool(conv2_2, 3, 3, 2, 2, padding='VALID', name='pool2_2')
        norm2_2 = lrn(pool2_2, 2, 2e-05, 0.75, name='norm2_2')
        
        # Layer 3 (conv-relu)
        # conv3 = conv(norm2, 3, 3, 384, 1, 1, name='conv3')
        conv3_2 = conv(norm2_2, 3, 3, 384, 1, 1, name='conv3_2')

        # Layer 4 (conv-relu)
        # conv4 = conv(conv3, 3, 3, 384, 1, 1, group=2, name='conv4')
        conv4_2 = conv(conv3_2, 3, 3, 384, 1, 1, group=2, name='conv4_2')
        
        # Layer 5 (conv-relu-pool)
        # conv5 = conv(conv4, 3, 3, 256, 1, 1, group=2, name='conv5')
        conv5_2 = conv(conv4_2, 3, 3, 256, 1, 1, group=2, name='conv5_2')
        
        pool5_2 = max_pool(conv5_2, 3, 3, 2, 2, padding='VALID', name='pool5_2')
        # Layer 6 (fc-relu-drop)
        fc6_2 = tf.reshape(pool5_2, [-1, 6*6*256])
        
        fc6r_2 = fc(fc6_2, 6*6*256, 4096,  name='fc6_2')
        # fc6r_2 = tf.nn.relu(fc6r_2)
        fc6_2 = dropout(fc6r_2, _dropout)
        

        return fc6r_2

    @staticmethod  
    def share_modal(feature1,feature2, _dropout, feature =True):           
        feat = tf.concat(0,[feature1, feature2])
        fc7 = fc(feat, 4096, 2048, relu=False, name='fc7')
        fc7r = dropout(tf.nn.relu(fc7), _dropout)
        result = fc(fc7r, 2048, 206, relu=False, name='fc8')
            
        if feature ==True:
            return fc7
        else:
            return result
            