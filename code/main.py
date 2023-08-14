import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch import nn
from dataset import AsrDataset,Phonemes,AsrDatasetAutoChoose
from model import LSTM_ASR
import numpy as np
from sklearn.model_selection import train_test_split
from test_script.crnn.models.crnn import CRNN
from ctc_decoder import beam_search
from config import debug,device,gBatchSize,test_debug,in_seq_length
import config as cf
import difflib
import matplotlib.pyplot as plt
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
    
    # old version when word_spellings is flattened
    target_lengths = torch.tensor([len(word_spellings[i]) for i in range(len(word_spellings))],dtype=torch.int32)
    word_spellings = torch.tensor([item for sublist in word_spellings for item in sublist],dtype=torch.int32)
    
    input_lengths = torch.tensor([int(len(feature) * cf.out_seq_length / cf.in_seq_length) for feature in features])
    def one_hot(indices, depth=256):
        one_hot_encode = np.zeros((len(indices),depth))
        one_hot_encode[np.arange(len(indices)),indices-1]=1
        return one_hot_encode
    
    if cf.feature_type == "quantized" and cf.use_vectorized_feature is False: # then use one-hot
        # old_version when input feature is one-hot 
        features = [one_hot(np.array(one_word_feature))  for one_word_feature in features]
        # features = [[np.eye(256)[np.array(one_frame_feature)-1] for one_frame_feature in one_word_feature]  for one_word_feature in features]

    padded_features = [np.pad(one_word_feature,((0,cf.in_seq_length-len(one_word_feature)),(0,0)),'constant',constant_values=0) for one_word_feature in features]
    padded_features = torch.stack([torch.tensor(f) for f in padded_features]).float()

    if debug: print(f'pad_feature shape: {padded_features.shape}')
    return padded_features, word_spellings, input_lengths, target_lengths
def untie_boosted_output(data,model,input_lengths,boosting_num):
    output = model(data,input_lengths)
    feature_len = output.shape[-1] // 3
    start_index = feature_len * boosting_num
    end_index = feature_len * (boosting_num + 1)
    output = output[:,:,start_index:end_index]
    output = output.transpose(0,1).contiguous()
    return output
def train(train_dataloader, model, ctc_loss, optimizer, epoch, train_loss, boosting_num=0):
    total_loss = 0
    num_batches = 0
    if not cf.use_boosting:
        for p in model.parameters():
            p.requires_grad = True
    else:
        for p in model.parameters():
            p.requires_grad = False
        for param in model.groups[boosting_num].parameters():
            param.requires_grad = True

    model = model.train().to(device)
    for batch_idx, (data, target, input_lengths, target_lengths) in enumerate(train_dataloader):
        if debug: print(f'\n\n======begin training batch {batch_idx}\ninput x model shape: {data.shape}')
        optimizer.zero_grad()
        data = data.to(device)
        target = target.to(device)

        if cf.use_boosting:
            output = untie_boosted_output(data,model,input_lengths,boosting_num)
        else:
            output = model(data,input_lengths).transpose(0,1).contiguous()# .requires_grad_(True) 

        if debug: print(f'model output shape (after transpose): {output.shape}')
        # input_lengths = torch.tensor([int(output.size(0))  for i in range(output.size(1))],dtype=torch.int32)
        if debug: print(f'ctcloss input lengths:\n {input_lengths}\nctcloss target lengths:\n {target_lengths}\n============== train batch {batch_idx} over\n\n')

        loss = ctc_loss(output, target, input_lengths, target_lengths)
        total_loss += loss.item()
        num_batches += 1
        loss.backward()
        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch+1},Batch: {batch_idx+1}/{len(train_dataloader)},loss:{loss.item()}')

        optimizer.step()
    
    avg_train_loss = total_loss / num_batches
    train_loss.append(avg_train_loss)


def main(training):
    #########
    test_loss, train_loss = [], []
    if cf.feature_type == "quantized":
        training_set = AsrDataset('data/split/clsp.trnscr.kept','data/split/clsp.trnlbls.kept','data/clsp.lblnames')
        train_dataset, val_dataset = train_test_split(training_set,test_size=0.2)
        # training_set = AsrDataset('data/clsp.trnscr','data/clsp.trnlbls','data/clsp.lblnames')
        test_set = AsrDataset('data/split/clsp.trnscr.held','data/split/clsp.trnlbls.held','data/clsp.lblnames')
    else:
        training_set = AsrDataset(scr_file='data/split/clsp.trnscr.kept',wav_scp='data/split/clsp.trnwav.kept',wav_dir='data/waveforms')
        # training_set = AsrDataset(scr_file='data/split/clsp.trnscr.kept.extend',wav_scp='data/split/clsp.trnwav.kept.extend',wav_dir='data/data_extend')
        test_set = AsrDataset(scr_file='data/split/clsp.trnscr.held',wav_scp='data/split/clsp.trnwav.held',wav_dir='data/waveforms')
    train_dataloader = DataLoader(training_set,batch_size=gBatchSize,shuffle=True,collate_fn=collate_fn)
    val_dataloader = DataLoader(test_set, batch_size=gBatchSize,shuffle=False,collate_fn=collate_fn)
    if cf.use_trainset_to_test:
        test_dataloader = DataLoader(training_set,batch_size=cf.test_batch_size,shuffle=False,collate_fn=collate_fn)
    else:
        test_dataloader = DataLoader(test_set,batch_size=cf.test_batch_size,shuffle=False,collate_fn=collate_fn)
    if cf.use_boosting:
        with open('data/clsp.trnscr','r') as file:
            words = list(set([line.strip() for line in file][1:]))
        autochoose_dataset = AsrDatasetAutoChoose(words,scr_file='data/split/clsp.trnscr.kept',wav_scp='data/split/clsp.trnwav.kept',wav_dir='data/waveforms')
        autochoose_dataloader = DataLoader(autochoose_dataset,batch_size=gBatchSize,shuffle=True,collate_fn=collate_fn)
        autochoose_testdataloader = test_dataloader
    model = LSTM_ASR(input_size=[cf.in_seq_length,cf.feature_size],output_size=[cf.out_seq_length,cf.gOutputSize],feature_type=cf.feature_type)

    # your can simply import ctc_loss from torch.nn
    loss_function = nn.CTCLoss(zero_infinity = True)

    # optimizer is provided
    optimizer = torch.optim.Adam(model.parameters(), lr=cf.gLr)
    # optimizer = torch.optim.SGD(model.parameters(),lr=1e-4,momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min',patience = 1000, factor=0.1)
    
    # Training
    autochoose_pool = []
    boosting_num = 0
    max_prob = []
    with open('data/clsp.trnscr','r') as file:
        words = list(set([line.strip() for line in file][1:]))
    past_word_dic = words
    if(training):
        if cf.use_pretrained:
            model.load_state_dict(torch.load(cf.pretrained_model_path))
            model.train()
            print('have load model')
        for epoch in range(cf.gNumEpoch):
            if not cf.use_boosting:
                train(train_dataloader, model, loss_function, optimizer, epoch, train_loss, boosting_num)
            else:
                train(autochoose_dataloader, model, loss_function, optimizer, epoch, train_loss, boosting_num)
            val_loss = compute_val_loss(val_dataloader, model, loss_function, boosting_num)
            scheduler.step(val_loss)
            test_loss.append(val_loss)
            if epoch % 1 == 0:
                torch.save(model.state_dict(),cf.model_path.split('.pth')[0]+f'epoch_{epoch}.pth')
            if debug: print('\n\n\n')
            if cf.use_boosting:
                if (epoch + 1) % 70 == 0:
                    if boosting_num < 2:
                        print(f'begin Boosting: {boosting_num + 1}')
                        with torch.no_grad():
                            for batch_idx, (data,target,input_lengths,output_lengths) in enumerate(autochoose_testdataloader):
                                data = data.to(device)
                                output = untie_boosted_output(data,model,input_lengths,boosting_num)
                                probabilities = compute_word_probabilities(output,past_word_dic,loss_function,input_lengths)
                                if max(probabilities) < 0.6:
                                    autochoose_pool.append(''.join(chr(i+96) for i in target if i != 0))
                                    autochoose_pool.append(past_word_dic[np.argmax(probabilities)])
                                max_prob.append(max(probabilities))
                        autochoose_pool = list(set(autochoose_pool))
                        if (len(past_word_dic) - len(autochoose_pool)) > 7: 
                            past_word_dic = autochoose_pool
                            print(f'words included: {autochoose_pool}')
                            autochoose_dataset = AsrDatasetAutoChoose(autochoose_pool,
                                                                    scr_file='data/split/clsp.trnscr.kept',wav_scp='data/split/clsp.trnwav.kept',wav_dir='data/waveforms')
                            autochoose_dataloader = DataLoader(autochoose_dataset,batch_size=int(gBatchSize)-boosting_num-1,shuffle=True,collate_fn=collate_fn)
                            autochoose_testdataloader = DataLoader(autochoose_dataset,batch_size=1,shuffle=False,collate_fn=collate_fn)
                            boosting_num += 1
                            autochoose_pool = []
                            optimizer = torch.optim.Adam(model.parameters(), lr=cf.gLr/(0.55*(boosting_num+1)))
                            print(f'included words num: {len(autochoose_pool)}\n end Boosting {boosting_num}')
        plt.plot(train_loss,label='train loss')
        plt.plot(test_loss, label='test loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title(cf.model_path)
        plt.savefig(cf.model_path.split('.pth')[0]+f'epoch_{cf.test_epoch_num}.pth'+'.png')
        plt.show()
    else:
        test_path = cf.test_model_path.split('.pth')[0]+f'epoch_{cf.test_epoch_num}.pth'
        if os.path.exists(test_path):        
            model.load_state_dict(torch.load(test_path))
            model = model.to(device)
            model.eval()
            accuracy = compute_accuracy(test_dataloader,model,decode)
            print('Test Accuracy: {:.2f}%'.format(accuracy * 100))


def compute_val_loss(dataloader, model, loss_function, boosting_num=0):
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch_idx,(data, target, input_lengths, target_lengths) in enumerate(dataloader):
            data = data.to(device)
            target = target.to(device)
            if cf.use_boosting:
                output = untie_boosted_output(data,model,input_lengths,boosting_num)
            else:
                output = model(data,input_lengths).transpose(0,1).contiguous()# .requires_grad_()
            # input_lengths = torch.tensor([int(output.size(0))  for i in range(output.size(1))],dtype=torch.int32)
            loss = loss_function(output,target,input_lengths,target_lengths)
            total_loss += loss.item()
    return total_loss / len(dataloader)


def beam_search_decode(output,beam_size,input_lengths):
    sequences = [[list(),1.0]]
    output = output[:int(input_lengths.cpu().item())].cpu()
    # old version when writing by myself
    # for row in output:
    #     all_candidates = []
    #     for i in range(len(sequences)):
    #         seq,score = sequences[i]
    #         for j in range(len(row)):
    #             candidate = [seq+[j], score * -row[j].cpu()]
    #             all_candidates.append(candidate)
    #     ordered = sorted(all_candidates, key=lambda tup:tup[1])
    #     sequences = ordered[:beam_size]
    # print(sequences[0][0])
    # return sequences[0][0]

    # new version when directly using library
    res = beam_search(output,[int_to_char(i) for i in range(25)])
    print(res)
    return res
    
def compute_word_probabilities(output, target_words, ctc_loss, input_lengths, phonemes_class=None):
    probabilities = []
    for target_word in target_words:
        if not cf.use_phoneme:
            target = torch.tensor([0] + [char_to_int(c) for c in target_word] + [0],dtype=torch.int32)
        else:
            target = torch.tensor([0] + [phonemes_class.phonemes_to_int[c] for c in target_word] + [0], dtype=torch.int32)
        loss = ctc_loss(output,target,input_lengths,torch.tensor([len(target_word)+2]))
        probability = torch.exp(-loss).item()
        probabilities.append(probability)
        
    probabilities = [p/sum(probabilities) for p in probabilities]
    return probabilities
def int_to_char(int_val):
    return chr(int_val + 96)
def char_to_int(char):
    return ord(char) - ord('a') + 1
def get_closest_word(decoded_output, words):

    decoded_word = ''.join(decoded_output)
    closest_word = difflib.get_close_matches(decoded_word,words,n=1,cutoff=0.3)
    return closest_word[0] if closest_word else "哇塞"

def decode(output, input_lengths, words, phonemes_class):
    #贪婪解码
    if test_debug: print(f'model output.shape:{output.shape}')

    # old version: find the largest
    output = torch.argmax(output,dim=-1).to('cpu')

    # find the 2nd largest
    # _, indices = torch.topk(output,2,dim=-1)
    # output = indices[:,:,1]
    if test_debug: print(f'output:{output}')

    decoded_output=[]
    blank_label= 0
    # for sequence in output:
    decoded_sequence = []
    previous_label = blank_label
    for label in output:
        if label!=blank_label and label != previous_label:
            if not cf.use_phoneme:
                decoded_sequence.append(int_to_char(label.item()))
            else:
                decoded_sequence.append(phonemes_class.int_to_phonemes[label.item()])
        previous_label = label
    decoded_output.append(decoded_sequence) if not cf.use_phoneme else decoded_output.append(''.join(decoded_sequence))
    decoded_output  = decoded_output[0]

    decoded_output = get_closest_word(decoded_output, words)
    
    return decoded_output

def compute_accuracy(dataloader, model, decode):
    correct = 0
    total = 0
    ctc_loss = nn.CTCLoss(zero_infinity=True)
    if cf.use_phoneme:
        phonemes_class = Phonemes('data/clsp.trnscr')
        words = phonemes_class.word_phonemes


    else: 
        phonemes_class = None
        with open('data/clsp.trnscr','r') as file:
            words = list(set([line.strip() for line in file][1:]))
    with torch.no_grad():
        for batch_idx,(data,target,input_lengths,output_lengths) in enumerate(dataloader):
            data = data.to(device)
            if cf.use_boosting:     
                decoded_output_ = 0 
                cross_group_probabilities = []
                for boost_cycle in range(3):
                    output = untie_boosted_output(data,model,input_lengths,boost_cycle)
                    probabilities = compute_word_probabilities(output, words, ctc_loss, input_lengths, phonemes_class)
                    cross_group_probabilities.append(probabilities)
                    if max(probabilities) > 0.4: # %$ 0.415:#0.415: # 0.42
                        probabilities = sorted(probabilities)
                        if probabilities[0] - probabilities[1] > 0.2: 
                            decoded_output_ = words[np.argmax(probabilities)]
                        break
                    else:
                        boost_cycle += 1
                if decoded_output_ == 0:
                    # result = np.sum(np.array(cross_group_probabilities),axis=0)
                    decoded_output_ = words[np.argmax(cross_group_probabilities[-1])]
                        
                        
                target = ''.join([int_to_char(t.item()) for t in target[1:-1]])
                print(f'decoded_output: {decoded_output_}, target:{target}, probability: {np.max(probabilities)}')

                if decoded_output_ == target:
                    correct+=1
                total+=1
            else:
                output = model(data,input_lengths)[0]
                # print(torch.argmax(output,dim=-1))
                probabilities = compute_word_probabilities(output, words, ctc_loss, input_lengths, phonemes_class)
                if cf.greedy_decode:
                    decoded_output = decode(output,input_lengths,words,phonemes_class)
                else:
                    decoded_output = beam_search_decode(output,10,input_lengths)
                if not cf.use_phoneme:
                    target = ''.join([int_to_char(t.item()) for t in target[1:-1]])
                    decoded_output_ = words[np.argmax(probabilities)]
                else:
                    target = ''.join([phonemes_class.int_to_phonemes[t.item()] for t in target[1:-1]])
                    # decoded_output_ = ''.join(words[np.argmax(probabilities)])
                # print(f'target:{target}')
                # old version when decoded_output is merged to 48 words
                
                print(f'decoded_output:{decoded_output}, max decoded_output: {decoded_output_}, target:{target}, probability: {np.max(probabilities)}')
                if decoded_output_ == target:
                    correct+=1
                total+=1
                # new version when decoded_output is original
                # for i in range(len(decoded_output)):
                #     if(i<len(target)):
                #         if decoded_output[i] == target[i]:
                #             correct+=1
                #     total+=1
    return correct / total

if __name__ == "__main__":
    main(training = cf.train_mode)