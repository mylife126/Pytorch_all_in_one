import os
import numpy as np
from PIL import Image
import argparse

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from models.ResNet import *
import time
import math

def create_parser():

    parser = argparse.ArgumentParser()
    
    # hypers
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--max_epoch", type=int, help="max number of epochs")
    parser.add_argument("--lr", type=float, help="learning rate")
    parser.add_argument("--num_classes", type=int, help="classification label classes")
    
    
    parser.add_argument("--step_per_epoch", type=int, help="steps in one epoch")   
    parser.add_argument("--max_step", type=int, help="max steps") 
    parser.add_argument("--global_step", type=int, help="used to set global step")
    parser.add_argument("--global_epoch", type=int, help="used to set global epochp")
    
    # paths
    parser.add_argument("--train_root", type=str, default='medium/', help="train folder root")
    parser.add_argument("--val_root", type=str, default='medium_dev/', help="val folder root")
    
    # opts
    parser.add_argument("--num_workers", type=int, default=2, help="number of processes in the loader")
    parser.add_argument("--device", type=str, help="gpu or cpu device")
    
    parser.add_argument("--progress_freq", type=int, help="print loss every prog freq")
    
    args = parser.parse_args()
    
    return args

def create_loader(args, shuffle=True):

    train_dataset = torchvision.datasets.ImageFolder(root=args.train_root, 
                                                           transform=torchvision.transforms.ToTensor())
    val_dataset = torchvision.datasets.ImageFolder(root=args.val_root, 
                                                           transform=torchvision.transforms.ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.num_workers)
        
    args.step_per_epoch = int(math.ceil(train_dataset.__len__() / args.batch_size))
    args.max_step = args.step_per_epoch * args.max_epoch
    args.num_classes = len(train_dataset.classes)
    print("Dataset has %d image folders, %d images in total" % (len(train_dataset.classes), train_dataset.__len__()))
    for idx, (inputs, labels) in enumerate(train_loader):
        print("input image shape:\t", inputs.shape)
        print("label shape:\t", labels.shape)
        break    
    
    return train_loader, val_loader

  
def param_init(m):
    """
    Initialization method in ResNet source code
    """
    
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

def train_epoch(args, model, train_loader, criterion, optimizer, start_time):
    model.train()
    model.to(args.device) 
    running_loss = 0
    total_predictions = 0.0
    correct_predictions = 0.0       
    for idx, (inputs, labels) in enumerate(train_loader):

        args.global_step += 1
        optimizer.zero_grad() 
        
        inputs = inputs.to(args.device)
        labels = labels.long().to(args.device)      
        outputs = model.forward(inputs)
        
        loss = criterion(outputs, labels)
        running_loss += loss.item() 
        one_loss = loss.item()

        loss.backward()
        optimizer.step()      
        
        
        _, predicted = torch.max(outputs.data, 1)
        total_predictions += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()        
        
        
        # compute time
        end_time = time.time()
        speed = (end_time - start_time) / args.global_step
        remain = args.max_step - args.global_step
        
        if args.global_step %  args.progress_freq == 1:
        
            print("epoch ", args.global_epoch)
            print("step ", args.global_step)
            print("training loss %0.5f" % (one_loss))
            print("remaining time %d m" %(int(math.ceil(remain / (speed * 60)))))
        
    running_loss /= len(train_loader)
    acc = (correct_predictions / total_predictions) * 100.0
    
    print("=" * 50)
    print("epoch %d training completed" % (args.global_epoch))
    print('training accuracy %0.2f' % (acc), '%')     
    print("training loss %0.5f" % (running_loss))    
    return running_loss
    
def val_epoch(args, model, val_loader, criterion):
    
    with torch.no_grad():
        model.eval()
        model.to(args.device)
        
        running_loss = 0
        total_predictions = 0.0
        correct_predictions = 0.0        
        
        for idx, (inputs, labels) in enumerate(val_loader):
            
            inputs = inputs.to(args.device)
            labels = labels.long().to(args.device)      
            outputs = model.forward(inputs)    
            
            _, predicted = torch.max(outputs.data, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            
            loss = criterion(outputs, labels).detach()
            running_loss += loss.item()            
        
        running_loss /= len(val_loader)
        acc = (correct_predictions / total_predictions) * 100.0
        print('testing loss: %0.5f' % (running_loss))
        print('testing accuracy %0.2f' % (acc), '%')
        print("=" * 50)
        return running_loss, acc       

   
def main():    
    # **** Preparations **** #
    
    # create arguments
    args = create_parser()
    # create loader
    train_loader, val_loader = create_loader(args)
    # create model
    model = ResNet18(num_classes=args.num_classes)
    print(model)
    # apply initialization
    model.apply(param_init)
    # create criterion
    criterion = nn.CrossEntropyLoss()
    # create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # create device
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
    print("=" * 50)
    print("Using %s" %(args.device))    
    
    # **** begin training and validating **** #
    args.global_step = 0
    args.global_epoch = 0    
    print("=" * 50)
    for k, v in args._get_kwargs():
        print(k, "=", v)
    print("=" * 50)
    
    start_time = time.time()
    for epoch in range(args.max_epoch):
        args.global_epoch += 1
        train_epoch(args, model, train_loader, criterion, optimizer, start_time)
        val_epoch(args, model, val_loader, criterion)
    
main()  
    
