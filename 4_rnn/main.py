import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import time
import argparse 

def create_parser():
    parser = argparse.ArgumentParser()
    # hypers
    parser.add_argument("--batch_size", type=int, help="batch size")
    parser.add_argument("--max_epoch", type=int, help="max epochs")

    # paths
    parser.add_argument("--train_data_path", type=str, help="train npy")
    parser.add_argument("--train_label_path", type=str, help="train label npy")
    parser.add_argument("--dev_data_path", type=str, help="dev npy")
    parser.add_argument("--dev_label_path", type=str, help="dev label npy")
    parser.add_argument("--test_data_path", type=str, help="test npy")

    # opts
    parser.add_argument("--num_workers", type=int, help="number of processes in the loader")

    # params
    parser.add_argument("--device", type=str, help="cpu or gpu")


    args = parser.parse_args()
    return args


class TrainDataset(Dataset):
    def __init__(self, device, data_path, label_path):
        self.data_path = data_path
        self.label_path = label_path
        self.device = device
        self.raw_utt, self.raw_label = self.loadnpy()
        self.info()
        self.utts = [torch.tensor(self.raw_utt[i]) for i in range(self.raw_utt.shape[0])]
        self.labels = [torch.tensor(self.raw_label[i]) for i in range(self.raw_label.shape[0])]

    def __len__(self):
        return len(self.utts)

    def __getitem__(self, index):
        return self.utts[index].to(self.device), self.labels[index].to(self.device)

    def loadnpy(self):
        data = np.load(self.data_path, encoding='bytes')
        label = np.load(self.label_path, encoding='bytes')
        return data, label
    
    def info(self):
        print("utt has shape\t", self.raw_utt.shape)
        print("utt[0] has shape\t", self.raw_utt[0].shape)
        print("utt[250] has shape\t", self.raw_utt[250].shape)
        print("label has shape\t", self.raw_label.shape)
        print("label[0] has shape\t", self.raw_label[0].shape)
        print("label[250] has shape\t", self.raw_label[250].shape)      

class TestDataset(Dataset):
    def __init__(self, device, data_path):
        self.data_path = data_path
        self.device = device
        self.raw_utt = self.loadnpy()
        self.info()
        self.utts = [torch.tensor(self.raw_utt[i]) for i in range(self.raw_utt.shape[0])]

    def __len__(self):
        return len(self.utts)

    def __getitem__(self, index):
        return self.utts[index].to(self.device)

    def loadnpy(self):
        data = np.load(self.data_path, encoding='bytes')
        return data
    
    def info(self):
        print("utt has shape\t", self.raw_utt.shape)
        print("utt[0] has shape\t", self.raw_utt[0].shape)
        print("utt[250] has shape\t", self.raw_utt[250].shape)  

def collate_utts(seq_list):
    # seq_list has length batch_size, each element is a tuple of (utts, labels)
    utts, labels = zip(*seq_list)
    lens = [len(seq) for seq in utts] # length of utts
    seq_order = sorted(range(len(lens)), key=lens.__getitem__, reverse=True) # index of biggest to smallest
    utts = [utts[i] for i in seq_order]
    labels = [labels[i] for i in seq_order]
    return utts, labels 

def collate_utts_test(seq_list):
    # seq_list has length batch_size, each element is a tuple of (utts, labels)
    utts = seq_list
    lens = [len(seq) for seq in utts] # length of utts
    seq_order = sorted(range(len(lens)), key=lens.__getitem__, reverse=True) # index of biggest to smallest
    utts = [utts[i] for i in seq_order]
    return utts

def create_loader(args):
    print("="*50, "dataset", "="*50)
    train_dataset = TrainDataset(args.device, args.train_data_path, args.train_label_path)
    dev_dataset = TrainDataset(args.device, args.dev_data_path, args.dev_label_path)
    test_dataset = TestDataset(args.device, args.test_data_path)
    print("="*50, "dataloader", "="*50)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers, 
                            drop_last=False, collate_fn = collate_utts)
    dev_loader = DataLoader(dev_dataset, shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers, 
                            drop_last=True, collate_fn = collate_utts)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers, 
                            drop_last=False, collate_fn = collate_utts_test)                            
    print("Train dataset has %d utts in total" % (train_dataset.__len__()))
    print("Dev dataset has %d utts in total" % (dev_dataset.__len__()))
    print("Test dataset has %d utts in total" % (test_dataset.__len__()))    
    for idx, (utts, labels) in enumerate(train_loader):
        print("batch size is %d" %len(utts))
        print("one utt has shape:\t", utts[2].shape)
        print("one label shape:\t", labels[0].shape)
        break     
    return train_loader, dev_loader, test_loader

class CTCModel(nn.Module):
    def __init__(self, phoneme_size, embed_size, hidden_size, nlayers):
        super(CTCModel,self).__init__()
        self.phoneme_size = phoneme_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.nlayers = nlayers
        self.embedding = nn.Linear(phoneme_size, embed_size) # Embedding layer (num_embeddings, embedding_dim)
        self.rnn = nn.LSTM(input_size = embed_size,hidden_size=hidden_size,num_layers=nlayers) # Recurrent network
        self.scoring = nn.Linear(hidden_size, phoneme_size) # Projection layer

    def forward(self, utts):
        # pad utts
        batch_size = len(utts)
        lens = [len(s) for s in utts] # lens of all utts (already sorted)
        padded = rnn.pad_sequence(utts) # padded utts (max_seq_len, batch, 40)
        padded_flatten = padded.view(-1, )

        print(padded.shape)

        
        

def train_epoch(args, model, train_loader, dev_loader, test_loader, optimizer):
    for idx, (utts, labels) in enumerate(train_loader):
        outputs = model.forward(utts)
        break



def main():
    # create parser 
    args = create_parser()
    # specify device
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using device %s" %args.device)
    # create dataset and dataloader 
    train_loader, dev_loader, test_loader = create_loader(args)
    # create model 
    print("="*50, "model", "="*50)
    model = CTCModel(40, 256, 256, 3)
    model.to(args.device)
    print(model)
    print("="*50, "model parameters", "="*50)
    for idx, para in enumerate(model.parameters()):
        print(idx, para.shape)
    # create optimizer 
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001, weight_decay=1e-6)
    
    # begin training 
    for epoch in range(args.max_epoch):
        print("="*50, "training epoch %d" %epoch, "="*50)
        train_epoch(args, model, train_loader, dev_loader, test_loader, optimizer)
        








main()