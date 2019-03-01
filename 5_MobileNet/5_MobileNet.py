import os
import numpy as np
from PIL import Image
import argparse

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

def create_parser():

    parser = argparse.ArgumentParser()
    
    # hypers
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    
    # paths
    parser.add_argument("--img_root", type=str, default='medium/', help="image folder root")
    
    
    # opts
    parser.add_argument("--num_workers", type=int, default=2, help="number of processes in the loader")
    
    args = parser.parse_args()
    
    return args

def create_loader(args, shuffle=True):
    
    
    imageFolder_dataset = torchvision.datasets.ImageFolder(root=args.img_root, 
                                                           transform=torchvision.transforms.ToTensor())
    imageFolder_dataloader = DataLoader(imageFolder_dataset, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.num_workers)
    print("Dataset has %d image folders, %d images in total" % (len(imageFolder_dataset.classes), imageFolder_dataset.__len__()))
    for idx, batch in enumerate(imageFolder_dataloader):
        print("input image shape:\t", batch[0].shape)
        print("label shape:\t", batch[1].shape)
        break    
    
    return imageFolder_dataloader

#def conv3x3(in_channel, out_channel, kernel_size=kernel_size, stride=stride):
#    return nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride,
#                     padding=1, bias=False)
#
class ResNetBlock(nn.Module):
    
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

class ResNet50(nn.Module):
    def __init__(self, channel_size, stride=1):
        super(ResNet50, self).__init__()
        
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        
        
        
        
        self.block = nn.Sequential(nn.Conv2d(in_channels=channel_size, out_channels=channel_size, 
                                             kernel_size=3, stride=stride, padding=1, bias=False),
                                   nn.BatchNorm2d(num_features=channel_size),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(in_channels=channel_size, out_channels=channel_size, 
                                             kernel_size=3, stride=stride, padding=1, bias=False),
                                   nn.BatchNorm2d(num_features=channel_size))
        self.logit_non_linear = nn.ReLU(inplace=True)

    def forward(self, x):
        output = x
        output = self.block(output)
        output = self.logit_non_linear(output + x)
        return output
    
#    
#
#
#def create_model():
#    print(1)
    
def main():    
    
    args = create_parser()
    loader = create_loader(args)
    

    
main()  
    
