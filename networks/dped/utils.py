from __future__ import division
import numpy as np
import os
import os.path
import time
from glob import glob

from random import shuffle
import cv2
import math
from ops import *

def load_img(image_path, mode = "RGB"):
    return cv2.imread(image_path)

def save_img(img, path):
    cv2.imwrite(path, img)
    return 0

def resize_img(x, shape):
    x = np.copy(x).astype(np.uint8)
    y = cv2.resize(x, shape)
    return y

def calc_PSNR(img1, img2):
    #assume RGB image
    target_data = np.array(img1, dtype=np.float64)
    ref_data = np.array(img2,dtype=np.float64)
    diff = ref_data - target_data
    diff = diff.flatten('C')
    rmse = math.sqrt(np.mean(diff ** 2.) )
    if rmse == 0:
        return 100
    else:
        return 20*math.log10(255.0/rmse)
    
def load_dataset(config):
    phone_list = sorted(glob(config.train_path_phone))
    dslr_list = sorted(glob(config.train_path_dslr))
    print("Dataset: %s, %d image pairs" %(config.dataset_name, len(phone_list)))
    start_time = time.time()
    dataset_phone = [cv2.imread(filename) for filename in phone_list]
    dataset_dslr = [cv2.imread(filename) for filename in dslr_list]
    print("%d image pairs loaded! setting took: %4.4fs" % (len(dataset_phone), time.time() - start_time))
    return dataset_phone, dataset_dslr

def get_batch(dataset_phone, dataset_dslr, config, start = 0):
    phone_batch = np.zeros([config.batch_size, config.patch_size, config.patch_size, config.channels], dtype = 'float32')
    dslr_batch = np.zeros([config.batch_size, config.patch_size, config.patch_size, config.channels], dtype = 'float32')

    for i in range(config.batch_size):
        index = np.random.randint(len(dataset_phone))
        phone_patch = dataset_phone[index]
        dslr_patch = dataset_dslr[index]

        # randomly flip, rotate patch (assuming that the patch shape is square)
        if config.augmentation == True:
            prob = np.random.rand()
            if prob > 0.5:
                phone_patch = np.flip(phone_patch, axis = 0)
                dslr_patch = np.flip(dslr_patch, axis = 0)
            prob = np.random.rand()
            if prob > 0.5:
                phone_patch = np.flip(phone_patch, axis = 1)
                dslr_patch = np.flip(dslr_patch, axis = 1)
            prob = np.random.rand()
            if prob > 0.5:
                phone_patch = np.rot90(phone_patch)
                dslr_patch = np.rot90(dslr_patch)
        #phone_batch[i,:,:,:] = phone_patch
        #dslr_batch[i,:,:,:] = dslr_patch
        phone_batch[i,:,:,:] = preprocess(phone_patch) # pre/post processing function is defined in ops.py
        dslr_batch[i,:,:,:] = preprocess(dslr_patch)
    return phone_batch, dslr_batch