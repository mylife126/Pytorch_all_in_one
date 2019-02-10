#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data.dataloader as dataloader
import torch.optim as optim

from torch.utils.data import TensorDataset
from torchvision import transforms
from torchvision.datasets import MNIST

import matplotlib.pyplot as plt
import time




# *** part1 load data *** #

# 1.1 download mnist 

# root for data save, train for training and testing ,download for download
# transfer to tensor
train = MNIST('./data', train=True, download=True, transform=transforms.ToTensor())

test = MNIST('./data', train=False, download=True, transform=transforms.ToTensor())

cuda = torch.cuda.is_available()


# 1.2 create dataloader
# define batch size and shuffle here
# elegant way to use dataloader_args

cuda = torch.cuda.is_available()
dataloader_args = dict(shuffle=True, batch_size=256,num_workers=4, pin_memory=True) if cuda \
                        else dict(shuffle=True, batch_size=32)
train_loader = dataloader.DataLoader(train, **dataloader_args) 

test_loader = dataloader.DataLoader(test, **dataloader_args)




# 1.3 good practice to know the shape #
train_data = train.train_data
train_data = train.transform(train_data.numpy())

print('[Train]')
print(' - Numpy Shape:', train.train_data.cpu().numpy().shape)
print(' - Tensor Shape:', train.train_data.size())
print(' - min:', torch.min(train_data))
print(' - max:', torch.max(train_data))
print(' - mean:', torch.mean(train_data))
print(' - std:', torch.std(train_data))
print(' - var:', torch.var(train_data))



# 1.4 also good practice to visualize it # 
fig1 = plt.figure()
plt.imshow(train.train_data.cpu().numpy()[0], cmap='gray')
plt.close()


# *** part2 construct MLP network *** #

# 2.1 MLP network #



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
            
            
# 2.2 Instantialize MLP model, set criterion,  optimizer and device
        
# Let's make a 3 layer mlp
mlp_model = MLPnet([784, 256, 10])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(mlp_model.parameters())
device = torch.device("cuda" if cuda else "cpu")
            
            
print(mlp_model)
        
# 2.3 Function for one epoch training

def Train_epoch(model, train_loader, criterion, optimizer, device="cpu"):
    
    # start training mode
    model.train()
    
    # specify device
    model.to(device)
    
    # initialize loss for current mini-batch
    running_loss = 0.0
    
    # cal time 
    start_time = time.time()
    
    # load mini-batch
    for batch_idx, (data, target) in enumerate(train_loader): 
        
        # for each mini-batch, remember to reset gradient!
        optimizer.zero_grad()   
        
        # put data into the device 
        data = data.to(device)
        target = target.long().to(device)

        # compute output
        outputs = model(data)
        
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
def Test_model(model, test_loader, criterion, device="cpu"):
    
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
            target = target.long().to(device)

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


# 2.5 main
n_epochs = 10
Train_loss = []
Test_loss = []
Test_acc = []

for i in range(n_epochs):
    train_loss = Train_epoch(mlp_model, train_loader, criterion, optimizer)
    test_loss, test_acc = Test_model(mlp_model, test_loader, criterion)
    Train_loss.append(train_loss)
    Test_loss.append(test_loss)
    Test_acc.append(test_acc)
    print('='*20)
        
# 2.6 Plot
fig2 = plt.figure()
plt.title('Test Loss')
plt.xlabel('Epoch Number')
plt.ylabel('Loss')
plt.plot(Test_loss)
fig3 = plt.figure()
plt.title('Test Accuracy')
plt.xlabel('Epoch Number')
plt.ylabel('Accuracy (%)')
plt.plot(Test_acc)