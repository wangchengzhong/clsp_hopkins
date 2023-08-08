import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch import nn
from dataset import AsrDataset
from model import LSTM_ASR
import numpy as np
from sklearn.model_selection import train_test_split
from test_script.crnn.models.crnn import CRNN
from ctc_decoder import beam_search
from config import debug,device,gBatchSize,test_debug,in_seq_length
import config as cf

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
    
    
    # new version when word_spelling is padded, still no use
    # target_lengths = torch.tensor([len(word_spellings[i])for i in range(len(word_spellings))])
    # word_spellings = torch.tensor([sublist + [0]*(15-len(sublist)) for sublist in word_spellings])
    # word_spellings = torch.tensor([item for sublist in word_spellings for item in sublist])
    # if debug: print(f'word spellings shape after padding: {word_spellings.shape}')
    
    # old version when word_spellings is flattened
    target_lengths = torch.tensor([len(word_spellings[i]) for i in range(len(word_spellings))],dtype=torch.int32)
    word_spellings = torch.tensor([item for sublist in word_spellings for item in sublist],dtype=torch.int32)# [torch.tensor(w) for w in word_spellings]
    
    input_lengths = torch.tensor([int(len(feature) * 60 / in_seq_length) for feature in features])
    if not cf.use_vectorized_feature: # then use one-hot
        # old_version when input feature is one-hot 
        features = [[np.eye(256)[np.array(one_frame_feature)-1] for one_frame_feature in one_word_feature]  for one_word_feature in features]
    
    padded_features = [np.pad(one_word_feature,((0,in_seq_length-len(one_word_feature)),(0,0)),'constant',constant_values=0) for one_word_feature in features]
    padded_features = torch.stack([torch.tensor(f) for f in padded_features]).float()

    # old version when using conv2d
    padded_features = padded_features.unsqueeze(1)

    if debug: print(f'pad_feature shape: {padded_features.shape}')
    return padded_features, word_spellings, input_lengths, target_lengths

def train(train_dataloader, model, ctc_loss, optimizer, epoch):
    for p in model.parameters():
        p.requires_grad = True
    model = model.train().to(device)
    for batch_idx, (data, target, input_lengths, target_lengths) in enumerate(train_dataloader):
        if debug: print(f'\n\n======begin training batch {batch_idx}\ninput x model shape: {data.shape}')
        optimizer.zero_grad()
        data = data.to(device)
        target = target.to(device)

        output = model(data,input_lengths).transpose(0,1).contiguous()# .requires_grad_(True) 
        if debug: print(f'model output shape (after transpose): {output.shape}')
        # input_lengths = torch.tensor([int(output.size(0))  for i in range(output.size(1))],dtype=torch.int32)
        if debug: print(f'ctcloss input lengths:\n {input_lengths}\nctcloss target lengths:\n {target_lengths}\n============== train batch {batch_idx} over\n\n')


        loss = ctc_loss(output, target, input_lengths, target_lengths)

        loss.backward()
        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch+1},Batch: {batch_idx+1}/{len(train_dataloader)},loss:{loss.item()}')

        optimizer.step()


def main(training):
    #########

    training_set = AsrDataset('split/clsp.trnscr.kept','split/clsp.trnlbls.kept','data/clsp.lblnames')
    train_dataset, val_dataset = train_test_split(training_set,test_size=0.2)
    # training_set = AsrDataset('data/clsp.trnscr','data/clsp.trnlbls','data/clsp.lblnames')
    test_set = AsrDataset('split/clsp.trnscr.held','split/clsp.trnlbls.held','data/clsp.lblnames')

    train_dataloader = DataLoader(training_set,batch_size=gBatchSize,shuffle=True,collate_fn=collate_fn)
    val_dataloader = DataLoader(test_set, batch_size=gBatchSize,shuffle=True,collate_fn=collate_fn)
    
    test_dataloader = DataLoader(test_set,batch_size=gBatchSize,shuffle=True,collate_fn=collate_fn)


    model = LSTM_ASR(input_size=[in_seq_length,256],output_size=[60,cf.gOutputSize])

    # your can simply import ctc_loss from torch.nn
    loss_function = nn.CTCLoss(zero_infinity = True)

    # optimizer is provided
    optimizer = torch.optim.Adam(model.parameters(), lr=cf.gLr)
    # optimizer = torch.optim.SGD(model.parameters(),lr=1e-4,momentum=0.9)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min',patience = 1000,factor=0.99999)
    
    # Training
    if(training):
        if cf.use_pretrained:
            model.load_state_dict(torch.load(cf.pretrained_model_path))
            model.train()
            print('have load model')
        for epoch in range(cf.gNumEpoch):
            train(train_dataloader, model, loss_function, optimizer, epoch)
            val_loss = compute_val_loss(val_dataloader, model, loss_function)
            # scheduler.step(val_loss)
            if epoch % 50 == 0:
                torch.save(model.state_dict(),cf.model_path.split('.pth')[0]+f'epoch_{epoch}.pth')
            if debug: print('\n\n\n')
    else:
        if os.path.exists(cf.test_model_path):
            model.load_state_dict(torch.load(cf.test_model_path))
            model.eval()
            model = model.to(device)
        for batch_idx,(data, target, input_lengths, target_lengths) in enumerate(test_dataloader):
            if debug: print(f'data: {data.shape}')
            data = data.to(device)
            output = model(data,input_lengths)# .requires_grad_(True)
            if test_debug: print(f'model output shape: {output.shape}')
            # decoded_output = decode(output)

        accuracy = compute_accuracy(test_dataloader,model,decode)
        print('Test Accuracy: {:.2f}%'.format(accuracy * 100))


def compute_val_loss(dataloader, model, loss_function):
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch_idx,(data, target, input_lengths, target_lengths) in enumerate(dataloader):
            data = data.to(device)
            target = target.to(device)
            output = model(data,input_lengths).transpose(0,1).contiguous()# .requires_grad_()
            # input_lengths = torch.tensor([int(output.size(0))  for i in range(output.size(1))],dtype=torch.int32)
            loss = loss_function(output,target,input_lengths,target_lengths)
            total_loss += loss.item()
    return total_loss / len(dataloader)
def beam_search_decode(output,beam_size):
    output = output[0]
    sequences = [[list(),1.0]]
    for row in output:
        all_candidates = []
        for i in range(len(sequences)):
            seq,score = sequences[i]
            for j in range(len(row)):
                candidate = [seq+[j],score*-row[j].cpu()]
                all_candidates.append(candidate)
        ordered = sorted(all_candidates, key=lambda tup:tup[1])
        sequences = ordered[:beam_size]
    print(sequences[0][0])
    return sequences[0][0]
def decode(output):
    #贪婪解码
    # print(f'original output shape: {output}')
    if test_debug: print(f'model output.shape:{output.shape}')
    # old version: find the largest
    output = torch.argmax(output,dim=-1)

    # find the 2nd largest
    # _, indices = torch.topk(output,2,dim=-1)
    # output = indices[:,:,1]
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
    decoded_output  = decoded_output[0][:-1]
    print(decoded_output)
    return decoded_output

def compute_accuracy(dataloader, model, decode):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx,(data,target,input_lengths,output_lengths) in enumerate(dataloader):
            data = data.to(device)
            
            output = model(data,input_lengths)
            # print(output.shape)
            
            # print(output)
            decoded_output = beam_search_decode(output,10)
            # print(decoded_output)
            for i in range(len(decoded_output)):
                if(i<len(target)):
                    if decoded_output[i] == target[i]:
                        correct+=1
                total+=1
    return correct/total

if __name__ == "__main__":
    main(training = cf.train_mode)