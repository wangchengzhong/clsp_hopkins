import torch
import os
train_mode = False

debug= False
test_debug= False

gNumEpoch= 1000
gBatchSize= 15
gLr = 1e-3
device= torch.device("cuda:0")

use_phoneme = False
use_vectorized_feature = False
gOutputSize = 26 if not use_phoneme else 43

feature_type = "quantized"
def get_feature_params(feature_type):
    assert feature_type in ['quantized', 'mfcc']
    if feature_type == "quantized":
        #                                #quantized#
        #      in_seq_length    out_seq_length   feature_size     hidden_size
        return 182,             90,              256,             93
    else:#                               #MFCC#
        #      in_seq_length    out_seq_length   feature_size     hidden_size
        return 85,              85,              40,              93
    
in_seq_length, out_seq_length, feature_size, hidden_size = get_feature_params(feature_type)

greedy_decode = True

folder_name = f'{feature_type}_output_{gOutputSize}_hs_{hidden_size}'
folder_path = os.path.join('checkpoint', folder_name)
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
model_path= f'{folder_path}/model_{feature_type}_hiddensize_{hidden_size}_output_{gOutputSize}_batch_{gBatchSize}_lr_{gLr}_vec_{1 if use_vectorized_feature else 0}.pth'
use_pretrained= False
pretrained_model_path= 'checkpoint/model_epoch1000_lr_0.005.pth'

use_trainset_to_test = False
test_batch_size = 1
test_epoch_num = 100
test_model_path= f'{folder_path}/model_{feature_type}_hiddensize_{hidden_size}_output_{gOutputSize}_batch_{gBatchSize}_lr_{gLr}_vec_{1 if use_vectorized_feature else 0}.pth'