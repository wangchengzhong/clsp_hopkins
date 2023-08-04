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
from sklearn.model_selection import train_test_split
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
    # input_lengths = torch.tensor([len(features[i])for i in range(len(features))])
    target_lengths = torch.tensor([len(word_spellings[i])for i in range(len(word_spellings))])
    
    word_spellings = torch.tensor([item for sublist in word_spellings for item in sublist])# [torch.tensor(w) for w in word_spellings]

    features = [[np.eye(256)[np.array(one_frame_feature)-1] for one_frame_feature in one_word_feature]  for one_word_feature in features]
    padded_features = [np.pad(one_word_feature,((0,256-len(one_word_feature)),(0,0)),'constant',constant_values=0) for one_word_feature in features]
    padded_features = torch.stack([torch.tensor(f) for f in padded_features]).double()
    padded_features = padded_features.unsqueeze(1)
    # print(f'feature:{features[0].shape},feature length {len(features)}\n')
    # padded_features = pad_sequence(features, batch_first=False)
    # if debug: print(f'target:{word_spellings}')
    # if debug: print(f'input_lengths: {input_lengths}')
    if debug: print(f'pad_feature shape:{padded_features.shape}')
    return padded_features, word_spellings, target_lengths

def train(train_dataloader, model, ctc_loss, optimizer,epoch):
    model.train()
    for batch_idx, (data, target, target_lengths) in enumerate(train_dataloader):
        if debug: print(f'input x model shape: {data.shape}')
        data = data.to(device)
        target = target.double().to(device)
        optimizer.zero_grad()
        output = model(data).transpose(0,1).log_softmax(2)
        input_lengths = torch.tensor([output.size(0) for _ in range(output.size(1))])
        if debug: print(input_lengths)
        #important：裁剪（？）
        # print(output.shape)
        # output = nn.utils.rnn.pack_padded_sequence(output,input_lengths,batch_first=True)

        # input_lengths = torch.full(size=(len(output[0]),),fill_value=len(output),dtype=torch.long)
        # print(input_lengths)
        # target = target.view(-1, target.size(1)).transpose(0,1)
        # print(data[0][3])
        # print(f'input_dim:{len(output)}\n target:{target.shape}\n output:{output.shape}\n\n')
        loss = ctc_loss(output, target, input_lengths, target_lengths)
        # print(loss.item())
        loss.backward()
        if batch_idx % 20 == 0:
            print(f'Epoch: {epoch+1},Batch: {batch_idx+1}/{len(train_dataloader)},loss:{loss.item()}')

        optimizer.step()

def decode(output):
    #贪婪解码
    # print(f'original output shape: {output}')
    # output = output.transpose(0,1)
    if test_debug: print(f'model output.shape:{output.shape}')
    output = torch.argmax(output,dim=-1)
    if test_debug: print(f'output:{output}')
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
        for batch_idx,(data,target,output_lengths) in enumerate(dataloader):
            data = data.to(device)
            
            output = model(data)
            # print(output.shape)
            output = output.log_softmax(2)
            # print(output)
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
    global device,debug,test_debug
    debug = False
    test_debug = False
    device = torch.device("cuda:0")
    
    training_set = AsrDataset('data/clsp.trnscr','data/clsp.trnlbls','data/clsp.lblnames')
    train_dataset, val_dataset = train_test_split(training_set,test_size=0.2)

    test_set = AsrDataset('data/clsp.trnscr','data/clsp.trnlbls','data/clsp.lblnames')
   
    train_dataloader = DataLoader(training_set,batch_size=5,shuffle=True,collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=5,shuffle=False,collate_fn=collate_fn)
    test_dataloader = DataLoader(test_set,batch_size=5,shuffle=True,collate_fn=collate_fn)


    model = LSTM_ASR(input_size=[256,256],output_size=[16,26])
    
    # your can simply import ctc_loss from torch.nn
    loss_function = nn.CTCLoss()

    # optimizer is provided
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Training
    model_path = 'checkpoint/model.pth'
    num_epochs = 50
    if(use_trained):
        model = model.double().to(device)
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path))
            model.eval()
    else:
        model = model.double().to(device)
        for epoch in range(num_epochs):
           
            train(train_dataloader, model, loss_function, optimizer,epoch)
            torch.save(model.state_dict(),model_path)

    for batch_idx,(data,target,target_lengths) in enumerate(test_dataloader):
        if debug: print(f'data: {data.shape}')
        data = data.to(device)
        output = model(data).log_softmax(2)
        decoded_output = decode(output)

    accuracy = compute_accuracy(test_dataloader,model,decode)
    print('Test Accuracy: {:.2f}%'.format(accuracy * 100))


if __name__ == "__main__":
    main(use_trained=False)
    # else:
    #     model.load_state_dict(torch.load('checkpoint/model.pth'))
    #     model.eval()