import numpy as np
import cv2
import tensorflow as tf
import random

import pdb
class Dataset:
    def __init__(self, train_color_list, train_thermal_list):
        # Load training images (path) and labels
        
        with open(train_color_list) as f:
            data_color_list = open(train_color_list, 'rt').read().splitlines()
            # Get full list of color image and labels
            self.train_color_image = [s.split(' ')[0] for s in data_color_list]
            self.train_color_label = [int(s.split(' ')[1]) for s in data_color_list]

        # Load testing images (path) and labels
        with open(train_thermal_list) as f:
            data_thermal_list = open(train_thermal_list, 'rt').read().splitlines()
            # Get full list of thermal image and labels
            self.train_thermal_image = [s.split(' ')[0] for s in data_thermal_list]
            self.train_thermal_label = [int(s.split(' ')[1]) for s in data_thermal_list]

        # Init params
        self.train_ptr = 0
        # self.test_ptr = 0
        self.train_size = len(self.train_color_label)
        self.crop_size = 227
        self.scale_size = 256
        # self.mean = np.array([104., 117., 124.]) # original
        self.mean = np.array([123.68, 116.779, 103.939]) # ours
        self.n_classes = 206
    
    def read_img(self, data_path):
        img = cv2.imread('../Dataset/' + data_path)
        h, w, c = img.shape
        assert c==3
        img = cv2.resize(img, (self.scale_size, self.scale_size))
        img = img.astype(np.float32)
        img -= self.mean
        shift_idx = np.random.permutation(int((self.scale_size-self.crop_size)))
        shift = shift_idx[0]
        img_crop = img[shift:shift+self.crop_size, shift:shift+self.crop_size, :]

        return img_crop
        
    def next_batch(self, batch_size, phase):
        # Get next batch of image (path) and labels
        if phase == 'train':
            batch_color_imgs    = np.ndarray([batch_size, self.crop_size, self.crop_size, 3])
            batch_thermal_imgs  = np.ndarray([batch_size, self.crop_size, self.crop_size, 3])

            batch_color_ids    = np.zeros((batch_size, 1))
            batch_thermal_ids  = np.zeros((batch_size, 1))

            # one shot labels
            batch_color_labels   = np.zeros((batch_size, self.n_classes))
            batch_thermal_labels = np.zeros((batch_size, self.n_classes))
            
            # random select batch_size identities
            idx = np.random.permutation(self.n_classes)
            batch_idx = idx[:batch_size]
            for i in xrange(batch_size):
                # random select one color image and one thermal image for each identity 
                tmp_label = batch_idx[i]
                
                # sample idx of color image
                pos = [k for k,v in enumerate(self.train_color_label) if v==tmp_label]
                sample_pos = random.randint(0,len(pos)-1)
                sample_color_idx = pos[sample_pos]

                # sample idx of thermal image
                pos = [k for k,v in enumerate(self.train_thermal_label) if v==tmp_label]
                sample_pos = random.randint(0,len(pos)-1)
                sample_thermal_idx = pos[sample_pos]
                
                batch_color_imgs[i]   = self.read_img(self.train_color_image[sample_color_idx])
                batch_thermal_imgs[i] = self.read_img(self.train_thermal_image[sample_thermal_idx])
                
                batch_color_ids[i]    = self.train_color_label[sample_color_idx]
                batch_thermal_ids[i]  = self.train_thermal_label[sample_thermal_idx]
                
                # one shot labels
                batch_color_labels[i][self.train_color_label[sample_color_idx]]     = 1
                batch_thermal_labels[i][self.train_thermal_label[sample_thermal_idx]] = 1
                
            batch_labels =np.vstack((batch_color_labels,batch_thermal_labels)) 
            # pdb.set_trace()
            return batch_color_imgs, batch_thermal_imgs, batch_labels, batch_color_ids, batch_thermal_ids
            
        else:
            return None, None