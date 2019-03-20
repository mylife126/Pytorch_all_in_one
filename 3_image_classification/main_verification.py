
import os
import numpy as np
from PIL import Image
import argparse
from tqdm import tqdm as tqdm
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
    parser.add_argument("--num_classes", type=int, default=2300, help="classification label classes")
    parser.add_argument("--num_val", type=int, help="number of validation files")
    parser.add_argument("--num_test", type=int, help="number of test files")
  
    # paths
    parser.add_argument("--val_txt_path", help="trial txt file path for validation")
    parser.add_argument("--val_data_path", help="path where imgs locate for validation")
    parser.add_argument("--test_txt_path", help="trial txt file path for test")
    parser.add_argument("--test_data_path", help="path where imgs locate for test")
    parser.add_argument("--checkpoint", type=str, help="checkpoint to restore training")
    parser.add_argument("--val_sim_path", help="trial txt file path for validation") 
    parser.add_argument("--real_val_sim_path", help="trial txt file path for validation")
    parser.add_argument("--test_sim_path", help="trial txt file path for validation") 
    # opts
    parser.add_argument("--num_workers", type=int, default=2, help="number of processes in the loader")
    parser.add_argument("--device", type=str, help="gpu or cpu device")
    parser.add_argument("--model", choices=["res18", "res34", "res50", "res101", "res152"])
    args = parser.parse_args()
    
    return args
class Veri_dataset(Dataset):

    def __init__(self, txt_path, data_path, mode):
        self.txt_path = txt_path
        self.data_path = data_path
        self.mode = mode
        self.img1_dic = {}
        self.img2_dic = {}
        self.label_dic = {}
        self.path_dic = {}
        self.get_index()
        print("Initialization complete!")
    def get_index(self):
        base = os.getcwd()
        self.data_path = os.path.join(base, self.data_path)
        tempfile = open(self.txt_path, "r")
        count = 0
        for line in tempfile:
            line = line.rstrip()
            templist = line.split(' ')
            save_src = templist[0] + " " + templist[1]
            # get full path
            templist[0] = os.path.join(self.data_path, templist[0])
            templist[1] = os.path.join(self.data_path, templist[1])
            self.img1_dic[count] = templist[0]
            self.img2_dic[count] = templist[1]
            self.path_dic[count] = save_src
            if self.mode == "val":
                self.label_dic[count] = int(templist[2])  
            count += 1  
    def __len__(self):
        return len(self.img1_dic)
    def __getitem__(self, index):
        """
        Return two trials, label and path
        """
        file1 = self.img1_dic[index]
        file2 = self.img2_dic[index]
        img1 = Image.open(file1)
        img1 = torchvision.transforms.ToTensor()(img1)
        img2 = Image.open(file2)
        img2 = torchvision.transforms.ToTensor()(img2)
        path = self.path_dic[index]
        if self.mode == "val":
            label = self.label_dic[index]       
            return path, img1, img2, label
        else:
            return path, img1, img2



def createDataset(args):
    val_dataset = Veri_dataset(args.val_txt_path, args.val_data_path, "val")
    test_dataset = Veri_dataset(args.test_txt_path, args.test_data_path, "test")
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, 
                              num_workers=args.num_workers, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, 
                              num_workers=args.num_workers, drop_last=False)
    print("Validation has %d pairs" % val_dataset.__len__())   
    print("Test has %d pairs" % test_dataset.__len__())  
    args.num_val = val_dataset.__len__()
    args.num_test = test_dataset.__len__()
    for idx, (path, img1, img2, labels) in enumerate(val_loader):
        assert img1.shape == img2.shape
        print("validation input image shape:\t", img1.shape)
        print("validation label shape:\t", labels.shape)
        break    
    for idx, (path, img1, img2) in enumerate(test_loader):
        print("test input image shape:\t", img1.shape)
        print("test label shape:\t", labels.shape)
        break    
    return val_loader, test_loader
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
def get_similarity(args, model, val_loader, test_loader):
    with torch.no_grad():
        model.eval()
        model.to(args.device)
        val_score = np.zeros((args.num_val, 2),dtype=object)
        real_score = np.zeros((args.num_val, 2),dtype=object)
        test_score = np.zeros((args.num_test, 2),dtype=object)
        cos = torch.nn.CosineSimilarity()
        # # validation 
        # count = 0 
        # for idx, (path, img1, img2, labels) in tqdm(enumerate(val_loader)):
        #     batch_size = img1.shape[0]
        #     img1 = img1.to(args.device)   
        #     img2 = img2.to(args.device)    
        #     output1, embeddings1 = model.forward(img1)  
        #     output2, embeddings2 = model.forward(img2)
        #     _, predict1 = torch.max(output1.data, 1)
        #     _, predict2 = torch.max(output2.data, 1)
        #     path = np.array(path)
        #     #print(path.shape)
        #     #print(predict1)
        #     #print(predict2)
        #     #print(embeddings1[0])
        #     #print(embeddings2[0])
        #     temp = cos(embeddings1, embeddings2)
        #     #print(labels)
        #     #print(temp.cpu())
        #     val_score[count: count + batch_size, 0] = path[:]
        #     val_score[count: count + batch_size, 1] = temp.cpu().data
        #     real_score[count: count + batch_size, 0] = path[:]
        #     real_score[count: count + batch_size, 1] = labels[:]
        #     count += batch_size

        # np.savetxt(args.val_sim_path, val_score, delimiter=',', fmt="%s")
        # np.savetxt(args.real_val_sim_path, real_score, delimiter=',', fmt="%s")

        # test
        count = 0 
        for idx, (path, img1, img2) in tqdm(enumerate(test_loader)):
            batch_size = img1.shape[0]
            img1 = img1.to(args.device)   
            img2 = img2.to(args.device)    
            output1, embeddings1 = model.forward(img1)  
            output2, embeddings2 = model.forward(img2)
            _, predict1 = torch.max(output1.data, 1)
            _, predict2 = torch.max(output2.data, 1)
            path = np.array(path)
            #print(path.shape)
            #print(predict1)
            #print(predict2)
            #print(embeddings1[0])
            #print(embeddings2[0])
            temp = cos(embeddings1, embeddings2)
            #print(labels)
            #print(temp.cpu())
            test_score[count: count + batch_size, 0] = path[:]
            test_score[count: count + batch_size, 1] = temp.cpu().data
            count += batch_size

        np.savetxt(args.test_sim_path, test_score, delimiter=',', fmt="%s")
def main():
    #**** Preparations **** #

    # create arguments
    args = create_parser()
    # create loader
    val_loader, test_loader = createDataset(args)
    # create model
    model = model_selection(args)
    print(model)
    # load parameters
    model.load_state_dict(torch.load(args.checkpoint))
    print("checkpoint loaded!")   
    # create devices
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    #args.device = torch.device("cpu") 
    print("=" * 50)
    print("Using %s" %(args.device)) 
    # get cos similarity score
    get_similarity(args, model, val_loader, test_loader)

main()
