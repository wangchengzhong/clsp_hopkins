#!/usr/bin/env python

import os
import sys
import string
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from dataset import AsrDataset
from model import LSTM_ASR
import numpy as np

def get_dimensions(lst):
    if isinstance(lst, list):
        return [len(lst)] + get_dimensions(lst[0]) if lst else []
    return []
def collate_fn(batch):
    """
    This function will be passed to your dataloader.
    It pads word_spelling (and features) in the same batch to have equal length.with 0.
    :param batch: batch of input samples
    :return: (recommended) padded_word_spellings, 
                           padded_features,
                           list_of_unpadded_word_spelling_length (for CTCLoss),
                           list_of_unpadded_feature_length (for CTCLoss)
    """
    features, word_spellings = zip(*batch)
    # print(word_spellings)
    target_lengths = torch.tensor([len(word_spellings[i])for i in range(len(word_spellings))])

    #pad word_spellings
    
    word_spellings = torch.tensor([item for sublist in word_spellings for item in sublist])# [torch.tensor(w) for w in word_spellings]

    padded_word_spellings = word_spellings # pad_sequence(word_spellings, batch_first=False)
    #pad features
    features = [torch.tensor(f) for f in features]
    # print(f'feature:{features[0].shape},feature length {len(features)}\n')
    padded_features = pad_sequence(features, batch_first=False)
    input_lengths = torch.tensor([len(f) for f in padded_features])
    # print(padded_features.shape)

    return padded_features, padded_word_spellings, input_lengths, target_lengths

def train(train_dataloader, model, ctc_loss, optimizer,epoch):
    model.train()
    for batch_idx, (data, target, input_lengths, target_lengths) in enumerate(train_dataloader):
        data = data.double().to(device)
        target = target.double().to(device)
        optimizer.zero_grad()
        output = model(data).log_softmax(-1)


        input_lengths = torch.full(size=(len(output[0]),),fill_value=len(output),dtype=torch.long)
        # print(input_lengths)
        # target = target.view(-1, target.size(1)).transpose(0,1)
        # print(data[0][3])
        # print(f'input_dim:{len(output)}\n target:{target.shape}\n output:{output.shape}\n\n')
        loss = ctc_loss(output,target,input_lengths,target_lengths)
        # print(loss.item())
        loss.backward()
        if batch_idx % 20 == 0:
            print(f'Epoch: {epoch+1},Batch: {batch_idx+1}/{len(train_dataloader)},loss:{loss.item()}')

        optimizer.step()
    
def decode(output):
    #贪婪解码
    # print(f'original output shape: {output}')
    # output = output.transpose(0,1)
    output = torch.argmax(output,dim=-1)
   
    # print(f'output shape: {output[3]}')
    decoded_output=[]
    blank_label= 0
    for sequence in output:
        decoded_sequence = []
        previous_label = blank_label
        for label in sequence:
            if label!=blank_label and label != previous_label:
                decoded_sequence.append(label.item())
            previous_label = label
        decoded_output.append(decoded_sequence)
    return decoded_output

def compute_accuracy(dataloader, model, decode):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx,(data,target,input_lengths, output_lengths) in enumerate(dataloader):
            data = data.to(device)
            output = model(data).log_softmax(-1)
            decoded_output = decode(output)
            # print(decoded_output)
            for i in range(len(decoded_output)):
                if(i<len(target)):
                    if decoded_output[i] == target[i]:
                        correct+=1
                total+=1
    return correct/total

def main(use_trained):
    #########
    global device
    device = torch.device("cuda:0")
    
    training_set = AsrDataset('data/clsp.trnscr','data/clsp.trnlbls','data/clsp.lblnames')
    test_set = AsrDataset('data/clsp.trnscr','data/clsp.trnlbls','data/clsp.lblnames')
   
    train_dataloader = DataLoader(training_set,batch_size=50,shuffle=True,collate_fn=collate_fn)
    test_dataloader = DataLoader(test_set,batch_size=50,shuffle=False,collate_fn=collate_fn)


    model = LSTM_ASR(input_size=256,output_size=26)
    model = model.double().to(device)
    # your can simply import ctc_loss from torch.nn
    loss_function = nn.CTCLoss(blank=0)

    # optimizer is provided
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    model_path = 'checkpoint/model.pth'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        model.eval()
    # Training
    if(False):
        num_epochs = 50
        for epoch in range(num_epochs):
            train(train_dataloader, model, loss_function, optimizer,epoch)
            torch.save(model.state_dict(),model_path)
    # else:
    #     model.load_state_dict(torch.load('checkpoint/model.pth'))
    #     model.eval()
    for batch_idx,(data,target,input_lengths,target_lengths) in enumerate(test_dataloader):
        data = data.to(device)
        output = model(data)
        decoded_output = decode(output)

    accuracy = compute_accuracy(test_dataloader,model,decode)
    print('Test Accuracy: {:.2f}%'.format(accuracy * 100))


if __name__ == "__main__":
    main(use_trained=True)
