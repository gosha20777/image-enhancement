# Copyright 2020 by Andrey Ignatov. All Rights Reserved.

from torch.utils.data import Dataset
import numpy as np
import cv2
import torch
import random


class LoadTrainData(Dataset):
    def __init__(self, raw_pathes, rgb_pathes, test=False):
        self.raw_pathes = raw_pathes
        self.rgb_pathes = rgb_pathes
        self.test = test
        if len(raw_pathes) != len(rgb_pathes):
            raise ValueError("raw_pathes and rgb_pathes must have same length")

    def __len__(self):
        return len(self.raw_pathes)

    def __getitem__(self, idx):
        raw_image = cv2.imread(self.raw_pathes[idx])
        raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
        raw_image = raw_image.astype(np.float32) / 255.0
        raw_image = raw_image.transpose((2, 0, 1))

        dslr_image = cv2.imread(self.rgb_pathes[idx])
        dslr_image = cv2.cvtColor(dslr_image, cv2.COLOR_BGR2RGB)
        dslr_image = dslr_image.astype(np.float32) / 255.0
        dslr_image = dslr_image.transpose((2, 0, 1))

        if self.test is False:
            raw_image, dslr_image = self._augment(raw_image, dslr_image)    
        raw_image = torch.from_numpy(raw_image)  
        dslr_image = torch.from_numpy(dslr_image)
        return raw_image, dslr_image

    def _augment(self, *imgs):
        hflip = random.random() < 0.5
        vflip = random.random() < 0.5
        rot90 = random.random() < 0.5
        def _augment_func(img, hflip, vflip, rot90):
            if hflip:   img = img[:, :, ::-1]
            if vflip:   img = img[:, ::-1, :]
            if rot90:   img = img.transpose(0, 2, 1) # CHW
            return np.ascontiguousarray(img)
        return (_augment_func(img, hflip, vflip, rot90) for img in imgs)
