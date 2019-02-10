#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import numpy as np
import os
import torch
import torch.utils.data.dataset as dataset
import torch.utils.data.dataloader as dataloader
import torchvision.transforms as transforms
from tqdm import tqdm as tqdm


def Loadnpy(npy_path):
    
    data = np.load(npy_path, encoding='bytes')
    
    return data

def Writenpy(npy_path, data):
    
    np.save(npy_path, data)


class Totensor(object):

    """Convert ndarrays in sample to Tensors."""

    # not init here
    def __call__(self, sample):
        return torch.from_numpy(sample)
        
class Padding_view(object):
    
    """
        
    The class will take in a tensor with shape (2 * pad_size + 1, init_frame_size)
    and flatten it.
          
    """
               
    def __call__(self, sample):
        return sample.view(1, -1)
        
        

class Single_npy_dataset(dataset):
    
    def __init__(self, npy_path, num_frame, num_frame_list, transform=None):
        
        super(Single_npy_dataset, self).__init__()
        
        
        self.num_frame = num_frame
        self.num_frame_list = num_frame_list
        self.data = Loadnpy(npy_path)
        self.transform = transform
        
    def __len__(self):
        
        return self.num_frame   
    
    def __getitem__(self, index):
        
        data = np.load(self.npy_path, encoding='bytes')
        
        utt_ind, frame_ind = self.find_utt(index)
        
        one_frame = data[utt_ind][frame_ind, :]
        
        if self.transform:
            one_frame = self.transform(sample)        
        
        
        return one_frame.astype('float32')
    
     
    def find_utt(self, index):
        
        for ind, item in enumerate(self.num_frame_list):
            
            new_index = index - item
            
            if new_index < 0:
                break
            
            index = new_index
            
        return ind, index
    
base = os.getcwd()

train_path = base + "/data/2_Utt/train.npy"
test_path = base + "/data/2_Utt/test.npy"
train_path_labels = base + "/data/2_Utt/train_labels.npy"
test_path_labels = base + "/data/2_Utt/test_labels.npy"

