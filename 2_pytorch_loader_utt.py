#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import numpy as np
import os
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm as tqdm
import argparse
import matplotlib.pyplot as plt
import time

# Part 0, set hyperparameters #

parser = argparse.ArgumentParser()
parser.add_argument("--pad_coe", type=int, default=5, help="concatenate k adjanct frames")
parser.add_argument("--batch_size", type=int, default=32, help="mini batch size")
parser.add_argument("--size_list", type=int, default=[0, 128, 138], 
                                                    help="num of units for each layer")

a = parser.parse_args()

# Part 1, design a loader for padding .npy data #

def Loadnpy(npy_path):
    
    data = np.load(npy_path, encoding='bytes')
    
    return data

def Writenpy(npy_path, data):
    
    np.save(npy_path, data)


class Totensor(object):

    """Convert ndarrays in sample to Tensors."""

    # not init here
    def __call__(self, sample):
        return torch.from_numpy(sample).float()
        
class Padding_view(object):
    
    """
        
    The class will take in a tensor with shape (2 * pad_size + 1, init_frame_size)
    and flatten it.
          
    """
               
    def __call__(self, sample):
        return sample.view(1, -1)
        
 

class Single_npy_dataset(Dataset):
    

    def __init__(self, x_path, y_path, transform, pad_coe=5):
        
        super(Single_npy_dataset, self).__init__()
        
        self.pad_coe = pad_coe
        self.xdata = Loadnpy(x_path)
        self.ydata = Loadnpy(y_path)
        self.frame_num, self.frame_list = self.count_frame()
        
        self.transformx = transform[0]
        self.transformy = transform[1]
        
    def __len__(self):
        
        return self.frame_num  
    
    def __getitem__(self, index):
        
        utt_ind, frame_ind = self.find_utt(index)
        row = self.xdata[utt_ind].shape[0]
        col = self.xdata[utt_ind].shape[1]
        
        # get x
        # for frame in the beginning or ending 
        if frame_ind < self.pad_coe :
            zero_pad = np.zeros((self.pad_coe - frame_ind, col))
            non_zero_pad = self.xdata[utt_ind][0: frame_ind + self.pad_coe + 1, :]
            samplex = np.concatenate((zero_pad, non_zero_pad), axis=0)
            
        elif frame_ind > row - self.pad_coe - 1:
            zero_pad = np.zeros((self.pad_coe - row + frame_ind + 1, col))
            non_zero_pad = self.xdata[utt_ind][frame_ind - self.pad_coe:, :]
            samplex = np.concatenate((non_zero_pad, zero_pad), axis=0)
            
        else:
            samplex = self.xdata[utt_ind][frame_ind - self.pad_coe: frame_ind + self.pad_coe + 1, :]
        
        assert samplex.shape[0] == 2 * self.pad_coe + 1
        
        # get y
        sampley = np.array([self.ydata[utt_ind][frame_ind]])
        
        # transform
        if self.transformx:
            samplex = self.transformx(samplex)   
            sampley = self.transformy(sampley)
        
        sample = list([samplex, sampley])
        
        return sample
    
    def count_frame(self):
        
        count = 0 
        count_list = []
        
        for utt_ind in range(self.ydata.shape[0]):
            
            utt = self.ydata[utt_ind]
            count += utt.shape[0]
            count_list.append(utt.shape[0])
        
        return count, count_list      
     
    def find_utt(self, index):
        
        for ind, item in enumerate(self.frame_list):
            
            new_index = index - item
            
            if new_index < 0:
                break
            
            index = new_index
            
        return ind, index
    
    
# Part 2, the MLP model # 
        


class MLPnet(nn.Module):
    
    """
    MLPNet 
    
    Parameters:
        
        size_list (list) -- list of int, num of units for each layer, len: nlayer + 1
        
        x (tensor) -- input tensor, shape (batch, width, width)
    
    # 
    """
    
    def __init__(self, size_list):
        
        super(MLPnet, self).__init__()
        
        self.size_list = size_list
        self.nlayer = len(size_list) - 1
        self.layer_list = []
        
        for layer in range(1, self.nlayer):
            
            self.layer_list.append(nn.Linear(self.size_list[layer - 1], self.size_list[layer]))
            self.layer_list.append(nn.ReLU())
            
        # The output layer doesn't need ReLU!
        self.layer_list.append(nn.Linear(self.size_list[self.nlayer - 1], self.size_list[self.nlayer]))
        
        # save everything
        self.net = nn.Sequential(*self.layer_list)
        
    def forward(self, x):
        
        # since x is image here, we need to flatten it 
        # -1 means we only know the column and let the computer to compute the row num
        x = x.view(-1, self.size_list[0])
        return self.net(x)
            
            

        
# 2.3 Function for one epoch training

def Train_epoch(model, train_loader, criterion, optimizer, device="cuda"):
    
    # start training mode
    model.train()
    
    # specify device
    model.to(device)
    
    # initialize loss for current mini-batch
    running_loss = 0.0
    
    # cal time 
    start_time = time.time()
    
    # load mini-batch
    for batch_idx, (inputs, targets) in enumerate(train_loader): 
        
        # for each mini-batch, remember to reset gradient!
        optimizer.zero_grad()   
        
        # put data into the device 
        inputs = inputs.to(device)
        target = targets.squeeze().long().to(device)

        # compute output
        outputs = model(inputs)
        
        # compute criterion
        loss = criterion(outputs, target)
        
        # add loss of one sample
        # tensor.item get a python number from a tensor containing a single value 
        running_loss += loss.item() 

        # backprop
        loss.backward()
        
        # update para
        optimizer.step()        
        
    end_time = time.time()
    
    running_loss /= len(train_loader)
    print('Training Loss: ', running_loss, 'Time: ',end_time - start_time, 's')
    return running_loss    

# 2.4 Function for testing 
def Test_model(model, test_loader, criterion, device="cuda"):
    
    # model.eval() will notify all your layers that you are in eval mode, that way, 
    # batchnorm or dropout layers will work in eval model instead of training mode.
    # torch.no_grad() impacts the autograd engine and deactivate it. It will reduce 
    # memory usage and speed up computations but you won’t be able to backprop 
    
    with torch.no_grad():
        
        model.eval()
        model.to(device)

        running_loss = 0.0
        total_predictions = 0.0
        correct_predictions = 0.0
        
        for batch_idx, (data, target) in enumerate(test_loader):   
            data = data.to(device)
            target = target.squeeze().long().to(device)

            outputs = model(data)

            _, predicted = torch.max(outputs.data, 1)
            total_predictions += target.size(0)
            correct_predictions += (predicted == target).sum().item()
            
            # we don't / or can't compute the derivative of loss
            loss = criterion(outputs, target).detach()
            running_loss += loss.item()


        running_loss /= len(test_loader)
        acc = (correct_predictions/total_predictions)*100.0
        print('Testing Loss: ', running_loss)
        print('Testing Accuracy: ', acc, '%')
        return running_loss, acc    



    
if __name__ == "__main__":


    base = os.getcwd()
    
    train_path = base + "/data/2_Utt/train.npy"
    test_path = base + "/data/2_Utt/test.npy"
    train_path_labels = base + "/data/2_Utt/train_labels.npy"
    test_path_labels = base + "/data/2_Utt/test_labels.npy"
                                    
    train_loader = DataLoader(Single_npy_dataset(train_path, train_path_labels,
                                                   [transforms.Compose([Totensor(), Padding_view()]),
                                                   transforms.Compose([Totensor()])],
                                                   pad_coe=a.pad_coe),
                                                    batch_size=a.batch_size, shuffle=True)
                                    
    test_loader = DataLoader(Single_npy_dataset(test_path, test_path_labels,
                                                   [transforms.Compose([Totensor(), Padding_view()]),
                                                   transforms.Compose([Totensor()])],
                                                   pad_coe=a.pad_coe),
                                                    batch_size=a.batch_size, shuffle=False)  
    
    # print info
    for i, (inputs, targets) in enumerate(train_loader):
        
        input_size = inputs.shape[2]
        targets = targets.squeeze()
        
        print('[Train]')
        print(" - input shape is: ", inputs.shape)
        print(" - target shape is: ", targets.shape)
        break
    
    a.size_list[0] = input_size
    mlp_model = MLPnet(a.size_list)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(mlp_model.parameters())
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")    

    print(mlp_model)
         
    
        # 2.5 main
    n_epochs = 10
    Train_loss = []
    Test_loss = []
    Test_acc = []
    
    for i in range(n_epochs):
        train_loss = Train_epoch(mlp_model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = Test_model(mlp_model, test_loader, criterion, device)
        Train_loss.append(train_loss)
        Test_loss.append(test_loss)
        Test_acc.append(test_acc)
        print('='*20)    
    
            
