import numpy as np
import cv2
import tensorflow as tf
import random


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
        
    def next_batch(self, batch_size, phase):
        # Get next batch of image (path) and labels
        if phase == 'train':            
            if self.train_ptr + batch_size < self.train_size:
                idx = np.random.permutation(self.train_size)
                color_paths  = self.train_color_image[self.train_ptr:self.train_ptr + batch_size]
                color_labels = self.train_color_label[self.train_ptr:self.train_ptr + batch_size]
                thermal_paths = self.train_thermal_image[self.train_ptr:self.train_ptr + batch_size/2]
                thermal_labels = self.train_thermal_label[self.train_ptr:self.train_ptr + batch_size/2]
                
                for i in xrange(batch_size/2):
                    thermal_paths.append(self.train_thermal_image[idx[i]])
                    thermal_labels.append(self.train_thermal_label[idx[i]])
                
                self.train_ptr += batch_size
            else:
                new_ptr = (self.train_ptr + batch_size)%self.train_size                
                color_paths  = self.train_color_image[self.train_ptr:] + self.train_color_image[:new_ptr]
                color_labels = self.train_color_label[self.train_ptr:] + self.train_color_label[:new_ptr]
                tmp_thermal_paths  = self.train_thermal_image[self.train_ptr:] + self.train_thermal_image[:new_ptr]
                tmp_thermal_labels = self.train_thermal_label[self.train_ptr:] + self.train_thermal_label[:new_ptr]
                thermal_paths = tmp_thermal_paths[:batch_size/2]
                thermal_labels = tmp_thermal_labels[:batch_size/2]
                idx = np.random.permutation(self.train_size)
                for i in xrange(batch_size/2):
                    thermal_paths.append(self.train_thermal_image[idx[i]])
                    thermal_labels.append(self.train_thermal_label[idx[i]])
                    
                self.train_ptr = new_ptr
        else:
            return None, None
        
        
        # Read color images
        color_imgs = np.ndarray([batch_size, self.crop_size, self.crop_size, 3])
        
        for i in xrange(len(color_paths)):
            # 
            img = cv2.imread('../Dataset/' + color_paths[i])
            h, w, c = img.shape
            assert c==3
            
            img = cv2.resize(img, (self.scale_size, self.scale_size))
            img = img.astype(np.float32)
            img -= self.mean
            shift_idx = np.random.permutation(int((self.scale_size-self.crop_size)))
            shift = shift_idx[0]
            img_crop = img[shift:shift+self.crop_size, shift:shift+self.crop_size, :]
            # img_crop = img[shift:shift+self.crop_size, shift:shift+self.crop_size, :]
            color_imgs[i] = img_crop

        # Expand labels
        batch_color_labels = np.zeros((batch_size, self.n_classes))
        for i in xrange(len(color_labels)):
            batch_color_labels[i][color_labels[i]] = 1
        
        # Read thermal images
        thermal_imgs = np.ndarray([batch_size, self.crop_size, self.crop_size, 3])
        
        for i in xrange(len(thermal_paths)):
            # 
            img = cv2.imread('../Dataset/' + thermal_paths[i])
            h, w, c = img.shape
            assert c==3
            
            img = cv2.resize(img, (self.scale_size, self.scale_size))
            img = img.astype(np.float32)
            img -= self.mean
            # shift = int((self.scale_size-self.crop_size)/2)
            # img_crop = img[shift:shift+self.crop_size, shift:shift+self.crop_size, :]
            shift_idx = np.random.permutation(int((self.scale_size-self.crop_size)))
            shift = shift_idx[0]
            img_crop = img[shift:shift+self.crop_size, shift:shift+self.crop_size, :]
            # img_crop = img[shift:shift+self.crop_size, shift:shift+self.crop_size, :]
            thermal_imgs[i] = img_crop
        
        # Expand labels
        batch_thermal_labels = np.zeros((batch_size, self.n_classes))
        for i in xrange(len(thermal_labels)):
            batch_thermal_labels[i][thermal_labels[i]] = 1
        
        batch_labels_2 =np.vstack((batch_color_labels,batch_thermal_labels))  
        
        batch_cross_labels  = np.dot( batch_color_labels,batch_thermal_labels.T)
        
        # cross labels
        cross_labels = np.zeros((batch_size, 1))
        for i in xrange(batch_size):
            if thermal_labels[i]==color_labels[i]:
                cross_labels[i] = 1

        return color_imgs, thermal_imgs, batch_labels_2,cross_labels

