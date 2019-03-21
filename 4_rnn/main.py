import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import time

import shakespeare_data as sh

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)

# # Data - refer to shakespeare_data.py for details
corpus = sh.read_corpus()
# print("First 203 characters...Last 50 characters")
# print("{}...{}".format(corpus[:203], corpus[-50:]))
# print("Total character count: {}".format(len(corpus)))
chars, charmap = sh.get_charmap(corpus)
charcount = len(chars)
# print("Unique character count: {}\n".format(len(chars)))
shakespeare_array = sh.map_corpus(corpus, charmap)
# print("shakespeare_array.shape: {}\n".format(shakespeare_array.shape))
# small_example = shakespeare_array[:17]
# print("First 17 characters as indices", small_example)
# print("First 17 characters as characters:", [chars[c] for c in small_example])
# print("First 17 character indices as text:\n", sh.to_text(small_example,chars))

# Dataset class. Transform raw text into a set of sequences of fixed length, and extracts inputs and targets
class TextDataset(Dataset):
    def __init__(self, text, seq_len = 200): 
        """
        text is a long long string
        """
        # clip
        n_seq = len(text) // seq_len
        text = text[:n_seq * seq_len]
        # transform origial text to 2d tensor, as a mini-batch
        self.data = torch.tensor(text).view(-1,seq_len)
    def __getitem__(self, i):
        # get one seq, split into first n-1 chars and last n-1 chars
        txt = self.data[i]
        return txt[:-1],txt[1:]
    def __len__(self):
        # number of mini-batch
        return self.data.size(0)

# Collate function. Transform a list of sequences into a batch. Passed as an argument to the DataLoader.
# Returns data on the format seq_len x batch_size
def collate(seq_list):
    # seq_list is a list of tuples(inputs, targets)
    # unsqueeze(1) change (199, ) to (199, 1)
    inputs = torch.cat([s[0].unsqueeze(1) for s in seq_list], dim=1) 
    targets = torch.cat([s[1].unsqueeze(1) for s in seq_list], dim=1)
    return inputs, targets

class CharLanguageModel(nn.Module):

    def __init__(self, vocab_size, embed_size, hidden_size, nlayers):
        super(CharLanguageModel,self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.nlayers = nlayers
        self.embedding = nn.Embedding(vocab_size, embed_size) # Embedding layer (num_embeddings, embedding_dim)
        self.rnn = nn.LSTM(input_size = embed_size,hidden_size=hidden_size,num_layers=nlayers) # Recurrent network
        self.scoring = nn.Linear(hidden_size, vocab_size) # Projection layer
        
    def forward(self, seq_batch): 
        """
        Input seq_batch has shape(seq_len, batch), like(199, 32)
        """
        batch_size = seq_batch.size(1)
        embed = self.embedding(seq_batch)   # input: LongTensor of arbitrary shape containing the indices to extract
                                            # output shape (*, embed_size), * is the input shape
        hidden = None
        output_lstm, hidden = self.rnn(embed, hidden)  # input: (seq_len, batch, input_size)
                                                       # output shape (seq_len, batch, hidden_size)
        
        print(output_lstm.shape)
        print(len(hidden))
        output_lstm_flatten = output_lstm.view(-1, self.hidden_size) # shape(seq_len * batch, hidden_size)
        output_flatten = self.scoring(output_lstm_flatten) # (seq_len * batch, vocab_size)
        return output_flatten.view(-1,batch_size,self.vocab_size) # (seq_len, batch, vocab_size)

def train_epoch(model, train_loader, val_loader, optimizer, criterion):
    
    model = model.to(DEVICE)
    start_time = time.time()
    for idx, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(DEVICE)
        targets = targets.to(DEVICE)
        outputs = model.forward(inputs)

    

def fix_length_main():

    # **** dataset **** #
    print("=" * 50, "dataset", "=" * 50)
    split = 5000000
    train_dataset = TextDataset(shakespeare_array[:split])
    val_dataset = TextDataset(shakespeare_array[split:])
    print("train dataset has length %d" %train_dataset.__len__()) # 200 * 2500
    print("val dataset has length %d" %val_dataset.__len__())
    # **** dataloader **** #
    print("=" * 50, "dataloader", "=" * 50)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=64, collate_fn = collate)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=64, collate_fn = collate, drop_last=True)  
    for idx, (inputs, targets) in enumerate(train_loader):
        print("inputs have size" , inputs.shape) # 199 * 64, not 64 * 199 due to the collate_fn
        print("targets have size", targets.shape) 
        break
    # **** model **** #
    model = CharLanguageModel(charcount, 256, 256, 3) # 
    # **** optimizer **** #
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-6)
    # **** criterion **** #
    criterion = nn.CrossEntropyLoss()
    # **** train **** #
    for epoch in range(3):
        print("training epoch %d" %epoch)
        train_epoch(model, train_loader, val_loader, optimizer, criterion)


fix_length_main()