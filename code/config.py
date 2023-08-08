import torch
import os
train_mode= True

debug= False
test_debug= False

gNumEpoch= 2000
gBatchSize= 15
gLr=  5e-3
device= torch.device("cuda:0")

use_phoneme = False
use_vectorized_feature = True

gOutputSize = 26 if not use_phoneme else 43
in_seq_length = 182
hidden_size = 5

folder_name = f'output_{gOutputSize}_hs_{hidden_size}'
folder_path = os.path.join('checkpoint', folder_name)
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
model_path= f'{folder_path}/model_hiddensize_{hidden_size}_output_{gOutputSize}_batch_{gBatchSize}_lr_{gLr}_vec_{1 if use_vectorized_feature else 0}.pth'
use_pretrained= False
pretrained_model_path= 'checkpoint/model_epoch1000_lr_0.005.pth'
test_model_path= 'checkpoint/model_epoch1000_lr_0.005.pth'