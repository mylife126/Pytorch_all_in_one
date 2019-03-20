import os
import numpy as np
from PIL import Image
import argparse

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import math
import glob as glob

from models.ResNet import *

def create_parser():

    parser = argparse.ArgumentParser()
    
    # hypers
    parser.add_argument("--batch_size", required=True, type=int, default=32, help="batch size")
    parser.add_argument("--max_epoch", required=True, type=int, help="max number of epochs")
    parser.add_argument("--lr", type=float, help="learning rate")
    parser.add_argument("--num_classes", type=int, help="classification label classes")
    parser.add_argument("--step_per_epoch", type=int, help="steps in one epoch")   
    parser.add_argument("--max_step", type=int, help="max steps") 
    parser.add_argument("--global_step", type=int, help="used to set global step")
    parser.add_argument("--global_epoch", type=int, help="used to set global epoch")
    parser.add_argument("--num_test", type=int, help="number of test files")


    
    # paths
    parser.add_argument("--train_root", type=str, default='medium/', help="train folder root")
    parser.add_argument("--val_root", type=str, default='medium_dev/', help="val folder root")
    parser.add_argument("--test_root", type=str, default='medium_test/', help="test folder root")
    parser.add_argument("--test_txt", type=str, help="text for test files")
    parser.add_argument("--save_dir", type=str, default="save", help="dir to save model or just params")
    parser.add_argument("--pred_dir", type=str, default="pred", help="dir to save prediction result")
    parser.add_argument("--checkpoint", type=str, help="checkpoint to restore training")
    
    
    # opts
    parser.add_argument("--num_workers", type=int, default=2, help="number of processes in the loader")
    parser.add_argument("--device", type=str, help="gpu or cpu device")
    parser.add_argument("--model", choices=["res18", "res34", "res50", "res101", "res152"])
    parser.add_argument("--progress_freq", type=int, help="print loss every prog freq")

    args = parser.parse_args()
    
    return args

class TrainDataset(Dataset):
    """
    designed for train and validation data
    Raw data structure:
    -root
      - class index(like 1, 2, ..., 4000)
          - xxx.png
    """
    def __init__(self, data_dir):
        # use hash map to pair image and class index
        self.n_class = 0
        self.data_dic = {}
        self.data_dir = data_dir
        self.parse()
        self.key_list = [key for key in self.data_dic.keys()]
    def parse(self): 
        base = os.getcwd()
        all_folder = glob.glob(os.path.join(base, self.data_dir + "/*"))
        self.n_class = len(all_folder)
        for folder in all_folder:
            label = float(folder.split('/')[-1])
            all_file = glob.glob(os.path.join(folder, "*.jpg"))
            for file in all_file:
                self.data_dic[file] = label
    def __len__(self):
        return len(self.data_dic)
    def __getitem__(self, index):
        file = self.key_list[index]
        img = Image.open(file)
        img = torchvision.transforms.ToTensor()(img)
        label = self.data_dic[file]
        return img, label           
class TestDataset(Dataset):
    # designed only for test data
    def __init__(self, data_dir, textfile):
         file_name = open(textfile, "r")
         self.file_list = []
         for line in file_name:
             line = line.rstrip()
             self.file_list.append(data_dir + "/" + line)
             
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img = Image.open(self.file_list[index])
        img = torchvision.transforms.ToTensor()(img)
        return img
    
def create_loader(args):
     
    train_dataset = TrainDataset(args.train_root)
    val_dataset = TrainDataset(args.val_root)
    test_dataset = TestDataset(args.test_root, args.test_txt)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                              num_workers=args.num_workers, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, 
                            num_workers=args.num_workers, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, 
                            num_workers=args.num_workers, drop_last=False)    
        
    args.step_per_epoch = int(math.ceil(train_dataset.__len__() / args.batch_size))
    args.max_step = args.step_per_epoch * args.max_epoch
    args.num_classes = train_dataset.n_class
    print("Train dataset has %d image folders, %d images in total" % (train_dataset.n_class, train_dataset.__len__()))
    print("Validation dataset has %d image folders, %d images in total" % (val_dataset.n_class, val_dataset.__len__()))
    print("Test dataset has %d images in total" % (test_dataset.__len__()))
    args.num_test = test_dataset.__len__()

    for idx, (inputs, labels) in enumerate(train_loader):
        print("input image shape:\t", inputs.shape)
        print("label shape:\t", labels.shape)
        break    
    
    return train_loader, val_loader, test_loader

  
def param_init(m):
    """
    Intialize with 
    """
    
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


def model_selection(args):
    
    if args.model == "res18":
        return ResNet18(num_classes=args.num_classes)
    elif args.model == "res34":
        return ResNet34(num_classes=args.num_classes)
    elif args.model == "res50":
        return ResNet50(num_classes=args.num_classes)
    elif args.model == "res101":
        return ResNet101(num_classes=args.num_classes)
    elif args.model == "res152":
        return ResNet152(num_classes=args.num_classes)
    else:
        raise ValueError("no available model founded.")

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
        outputs, _ = model.forward(inputs)
        
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
            print("remaining time %d m" %(int(math.ceil(remain * speed / 60))))
        
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
            outputs, _ = model.forward(inputs)    
            
            _, predicted = torch.max(outputs.data, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            
            loss = criterion(outputs, labels).detach()
            running_loss += loss.item()            
        
        running_loss /= len(val_loader)
        acc = (correct_predictions / total_predictions) * 100.0
        print('validation loss: %0.5f' % (running_loss))
        print('validation accuracy %0.2f' % (acc), '%')
        print("=" * 50)
        return running_loss, acc       

def test_epoch(args, model, test_loader):
         
    pred_all = np.zeros((args.num_test, 2))    
    with torch.no_grad():
        model.eval()
        model.to(args.device)
        count = 0
        for idx, inputs in enumerate(test_loader):

            batch_size = inputs.shape[0]
            inputs = inputs.to(args.device)    
            outputs,_ = model.forward(inputs)    
            _, predicted = torch.max(outputs.data, 1)
            pred_all[count: count + batch_size, 0] = np.arange(count, count + batch_size)
            pred_all[count: count + batch_size, 1] = predicted.cpu().numpy()
            count += batch_size
          
        pred_path = os.path.join(os.getcwd(), args.pred_dir + "/pred%d.csv" % (args.global_epoch))
        np.savetxt(pred_path, pred_all, delimiter=',')
        print("Saving prediciton complete")        
        return predicted    

def main():
    #**** Preparations **** #
    base = os.getcwd()
    # create arguments
    args = create_parser()
    # create dataset and loader
    train_loader, val_loader, test_loader = create_loader(args)
    # create model
    model = model_selection(args)
    print(model)
    # apply initialization
    model.apply(param_init)
    # apply checkpoint, if applicable
    if args.checkpoint is not None:
        model.load_state_dict(torch.load(args.checkpoint))
        print("checkpoint loaded!")
    # create criterion
    criterion = nn.CrossEntropyLoss()
    # create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    # create devices
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
    print("=" * 50)
    print("Using %s" %(args.device))    
    # create paths
    if not os.path.exists(os.path.join(base, args.save_dir)):
        os.makedirs(os.path.join(base, args.save_dir))    
    if not os.path.exists(os.path.join(base, args.pred_dir)):
        os.makedirs(os.path.join(base, args.pred_dir))    
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
        test_epoch(args, model, test_loader)
        scheduler.step()
        
        # save model params every epoch
        param_path = os.path.join(args.save_dir, "para_epoch{}.pkl".format(args.global_epoch))
        torch.save(model.state_dict(), param_path)
main()       
    
