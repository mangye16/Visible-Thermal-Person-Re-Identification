import tensorflow as tf
import numpy as np
import sys
from network import *
import pdb

class Model:
    @staticmethod 
    def alexnet_visible(X1, _dropout, feature=False):
        # TODO weight decay loss tern
        net_data = np.load('../../bvlc_alexnet.npy').item()
        
        # Layer 1 (conv-relu-pool-lrn)
        # conv1_1 = conv(X1, 11, 11, 96, 4, 4, padding='VALID', name='conv1_1')
        with tf.name_scope('conv1_1') as scope:
            kernel = tf.Variable(net_data['conv1'][0], name='weights')
            conv = tf.nn.conv2d(X1, kernel, [1, 4, 4, 1], padding='VALID')
            biases = tf.Variable(net_data['conv1'][1],
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            conv1_1 = tf.nn.relu(out, name=scope)
            
        pool1_1 = max_pool(conv1_1, 3, 3, 2, 2, padding='VALID', name='pool1_1')
        norm1_1 = lrn(pool1_1, 2, 2e-05, 0.75, name='norm1_1')
        
        # Layer 2 (conv-relu-pool-lrn)
        # conv2 = conv(norm1, 5, 5, 256, 1, 1, group=2, name='conv2')
        with tf.name_scope('conv2_1') as scope:
            kernel = tf.Variable(net_data['conv2'][0], name='weights')
            group = 2
            convolve = lambda i, k: tf.nn.conv2d(i, k, [1, 1, 1, 1], padding='SAME')
            input_groups = tf.split(3, group, norm1_1)
            kernel_groups = tf.split(3, group, kernel)
            output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
            # Concatenate the groups
            conv = tf.concat(3, output_groups)
            biases = tf.Variable(net_data['conv2'][1],
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            conv2_1 = tf.nn.relu(out, name=scope)
            
        pool2_1 = max_pool(conv2_1, 3, 3, 2, 2, padding='VALID', name='pool2_1')
        norm2_1 = lrn(pool2_1, 2, 2e-05, 0.75, name='norm2_1')
        
        # Layer 3 (conv-relu)
        # conv3 = conv(norm2, 3, 3, 384, 1, 1, name='conv3')
        with tf.name_scope('conv3_1') as scope:
            kernel = tf.Variable(net_data['conv3'][0], name='weights')
            conv = tf.nn.conv2d(norm2_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(net_data['conv3'][1],
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            conv3_1 = tf.nn.relu(out, name=scope)
        
        # Layer 4 (conv-relu)
        # conv4 = conv(conv3, 3, 3, 384, 1, 1, group=2, name='conv4')
        with tf.name_scope('conv4_1') as scope:
            kernel = tf.Variable(net_data['conv4'][0], name='weights')
            group = 2
            convolve = lambda i, k: tf.nn.conv2d(i, k, [1, 1, 1, 1], padding='SAME')
            input_groups = tf.split(3, group, conv3_1)
            kernel_groups = tf.split(3, group, kernel)
            output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
            # Concatenate the groups
            conv = tf.concat(3, output_groups)
            biases = tf.Variable(net_data['conv4'][1],
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            conv4_1 = tf.nn.relu(out, name=scope)
        
        # Layer 5 (conv-relu-pool)
        # conv5 = conv(conv4, 3, 3, 256, 1, 1, group=2, name='conv5')
        with tf.name_scope('conv5_1') as scope:
            kernel = tf.Variable(net_data['conv5'][0], name='weights')

            group = 2
            convolve = lambda i, k: tf.nn.conv2d(i, k, [1, 1, 1, 1], padding='SAME')
            input_groups = tf.split(3, group, conv4_1)
            kernel_groups = tf.split(3, group, kernel)
            output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
            # Concatenate the groups
            conv = tf.concat(3, output_groups)

            biases = tf.Variable(net_data['conv5'][1],
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            conv5_1 = tf.nn.relu(out, name=scope)
        
        pool5_1 = max_pool(conv5_1, 3, 3, 2, 2, padding='VALID', name='pool5_1')
        # Layer 6 (fc-relu-drop)
        fc6_1 = tf.reshape(pool5_1, [-1, 6*6*256])
        
        # fc6 = fc(fc6, 6*6*256, 4096, name='fc6')
        # fc6 = dropout(fc6, _dropout)
        
        with tf.name_scope('fc6_1') as scope:
            fc6w_1 = tf.Variable(net_data['fc6'][0], name='weights')
            fc6b_1 = tf.Variable(net_data['fc6'][1],
                                 trainable=True, name='biases')
            fc6r_1 = tf.nn.bias_add(tf.matmul(fc6_1, fc6w_1), fc6b_1)
            # fc6r_1 = tf.nn.relu_layer(fc6_1, fc6w_1, fc6b_1, name=scope)
            
        fc6r_1 = tf.contrib.layers.batch_norm(fc6r_1)
        
        return fc6r_1
        
    @staticmethod  
    def alexnet_thermal(X2, _dropout, feature=False):
        # TODO weight decay loss tern
        net_data = np.load('../../bvlc_alexnet.npy').item()
        
        # Layer 1 (conv-relu-pool-lrn)
        # conv1_1 = conv(X1, 11, 11, 96, 4, 4, padding='VALID', name='conv1_1')
        with tf.name_scope('conv1_2') as scope:
            kernel = tf.Variable(net_data['conv1'][0], name='weights')
            conv = tf.nn.conv2d(X2, kernel, [1, 4, 4, 1], padding='VALID')
            biases = tf.Variable(net_data['conv1'][1],
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            conv1_2 = tf.nn.relu(out, name=scope)
            
        pool1_2 = max_pool(conv1_2, 3, 3, 2, 2, padding='VALID', name='pool1_2')
        norm1_2 = lrn(pool1_2, 2, 2e-05, 0.75, name='norm1_2')
        
        # Layer 2 (conv-relu-pool-lrn)
        # conv2 = conv(norm1, 5, 5, 256, 1, 1, group=2, name='conv2')
        with tf.name_scope('conv2_2') as scope:
            kernel = tf.Variable(net_data['conv2'][0], name='weights')
            group = 2
            convolve = lambda i, k: tf.nn.conv2d(i, k, [1, 1, 1, 1], padding='SAME')
            input_groups = tf.split(3, group, norm1_2)
            kernel_groups = tf.split(3, group, kernel)
            output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
            # Concatenate the groups
            conv = tf.concat(3, output_groups)
            biases = tf.Variable(net_data['conv2'][1],
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            conv2_2 = tf.nn.relu(out, name=scope)
            
        pool2_2 = max_pool(conv2_2, 3, 3, 2, 2, padding='VALID', name='pool2_2')
        norm2_2 = lrn(pool2_2, 2, 2e-05, 0.75, name='norm2_2')
        
        # Layer 3 (conv-relu)
        # conv3 = conv(norm2, 3, 3, 384, 1, 1, name='conv3')
        with tf.name_scope('conv3_2') as scope:
            kernel = tf.Variable(net_data['conv3'][0], name='weights')
            conv = tf.nn.conv2d(norm2_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(net_data['conv3'][1],
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            conv3_2 = tf.nn.relu(out, name=scope)
        
        # Layer 4 (conv-relu)
        # conv4 = conv(conv3, 3, 3, 384, 1, 1, group=2, name='conv4')
        with tf.name_scope('conv4_2') as scope:
            kernel = tf.Variable(net_data['conv4'][0], name='weights')

            group = 2
            convolve = lambda i, k: tf.nn.conv2d(i, k, [1, 1, 1, 1], padding='SAME')
            input_groups = tf.split(3, group, conv3_2)
            kernel_groups = tf.split(3, group, kernel)
            output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
            # Concatenate the groups
            conv = tf.concat(3, output_groups)
            biases = tf.Variable(net_data['conv4'][1],
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            conv4_2 = tf.nn.relu(out, name=scope)
        
        # Layer 5 (conv-relu-pool)
        # conv5 = conv(conv4, 3, 3, 256, 1, 1, group=2, name='conv5')
        with tf.name_scope('conv5_2') as scope:
            kernel = tf.Variable(net_data['conv5'][0], name='weights')
            group = 2
            convolve = lambda i, k: tf.nn.conv2d(i, k, [1, 1, 1, 1], padding='SAME')
            input_groups = tf.split(3, group, conv4_2)
            kernel_groups = tf.split(3, group, kernel)
            output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
            # Concatenate the groups
            conv = tf.concat(3, output_groups)
            biases = tf.Variable(net_data['conv5'][1],
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            conv5_2 = tf.nn.relu(out, name=scope)
        
        pool5_2 = max_pool(conv5_2, 3, 3, 2, 2, padding='VALID', name='pool5_2')
        # Layer 6 (fc-relu-drop)
        fc6_2 = tf.reshape(pool5_2, [-1, 6*6*256])
        
        # fc6 = fc(fc6, 6*6*256, 4096, name='fc6')
        # fc6 = dropout(fc6, _dropout)
        
        with tf.name_scope('fc6_2') as scope:
            fc6w_2 = tf.Variable(net_data['fc6'][0], name='weights')
            fc6b_2 = tf.Variable(net_data['fc6'][1],
                                 trainable=True, name='biases')
            fc6r_2 = tf.nn.bias_add(tf.matmul(fc6_2, fc6w_2), fc6b_2)
            # fc6r_2 = tf.nn.relu_layer(fc6_2, fc6w_2, fc6b_2, name=scope)
        # fc6r_2 = tf.nn.l2_normalize(fc6r_2, 0, epsilon=1e-12)    
        fc6r_2 = tf.contrib.layers.batch_norm(fc6r_2)
        return fc6r_2
        
    @staticmethod  
    def modal_embedding(feature1, feature2, _dropout, feature_dim = 1024, feature=False):
    
        feat = tf.concat(0,[feature1,feature2])
        feat = dropout(feat,_dropout)
        feat = tf.nn.relu(feat)
        fc7 = fc(feat, 4096, feature_dim, name='fc7')
        fc7r = dropout(tf.nn.relu(fc7), _dropout)
        result = fc(fc7r, feature_dim, 206, relu=False, name='fc8')
        
        if feature==True:
            return fc7
        else:
            return fc7, result
