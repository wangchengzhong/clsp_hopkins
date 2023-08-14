import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.init as init
import config as cf
if cf.use_transformer:
    from torch.nn import TransformerEncoder, TransformerEncoderLayer
use_gru = False
class CNN_LSTM_Group(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers,output_size,feature_type="mfcc"):
        super(CNN_LSTM_Group,self).__init__()
        self.feature_type = feature_type
        self.into_lstm_seq_length = 40
        self.output_size = output_size

        self.conv1 = nn.Conv2d(5,1,kernel_size=3,stride=1,padding=1)
        self.bn1 = nn.BatchNorm2d(1)
        self.lstm = nn.LSTM(input_size=self.into_lstm_seq_length,hidden_size=hidden_size,num_layers=num_layers,batch_first=True,bidirectional=True)
        self.fc = nn.Linear(in_features=hidden_size*2, out_features=output_size[1])
    
    def forward(self, batch_features, input_lengths):
        batch_features = batch_features.view(batch_features.shape[0],batch_features.shape[1],5,-1).transpose(1,2)
        x = self.conv1(batch_features)
        x = self.bn1(x)
        x = F.relu(x)
        x = x.squeeze(1)
        x = nn.utils.rnn.pack_padded_sequence(x,input_lengths,batch_first=True,enforce_sorted=False)
        x,_ = self.lstm(x)
        x,_ = nn.utils.rnn.pad_packed_sequence(x,batch_first=True,padding_value=0)

        if x.size(1)<self.output_size[0]:
            padding = torch.zeros((x.size(0),self.output_size[0]-x.size(1),x.size(2))).to(x.device)
            x = torch.cat([x,padding],dim=1)
        x = x.view(x.shape[0] * x.shape[1], -1)
        x = self.fc(x)
        x = x.view(-1,self.output_size[0],self.output_size[1])
        x = F.log_softmax(x,dim=-1)
        return x
    
class LSTM_ASR(torch.nn.Module):
    def __init__(self, input_size=[cf.in_seq_length,256], hidden_size=cf.hidden_size, num_layers=cf.num_layers,
                 output_size=[62,26],feature_type="quantized"):
        super().__init__()
        self.output_size = output_size
        self.feature_type = feature_type
        if cf.use_transformer:
            encoder_layers = TransformerEncoderLayer(d_model=int(hidden_size/2),nhead=8)
            self.transformer_encoder = TransformerEncoder(encoder_layer=encoder_layers,num_layers=6)
        # cancel conv2d
        if feature_type == "quantized":
            self.conv1 = nn.Conv2d(1, 1, kernel_size=3,stride=2)
            self.bn1 = nn.BatchNorm2d(1)

            # when using mfcc
            self.into_lstm_seq_length = 127 # 19
            if not use_gru:
                self.lstm = nn.LSTM(input_size=self.into_lstm_seq_length,hidden_size=hidden_size,num_layers=num_layers,batch_first=True,bidirectional=True)
            else:
                self.gru = nn.GRU(input_size=self.into_lstm_seq_length,hidden_size=hidden_size,num_layers=num_layers,batch_first=True,bidirectional=True)
                # self.dropout = nn.Dropout(0.5)
            self.fc = nn.Linear(in_features = hidden_size * 2, out_features = output_size[1])
        else:
            # #old version when conv2d in channel = 1
            
            # #old version when not using conv2d in mfcc mode
            # # self.into_lstm_seq_length = 120
            # self.conv1 = nn.Conv2d(1, 1, kernel_size=3,stride=2,padding=1)
            # self.bn1 = nn.BatchNorm2d(1)

            # # when using mfcc
            # self.into_lstm_seq_length = 100# 20 60 #59*2 # 19

            
            self.group = CNN_LSTM_Group(input_size=input_size,hidden_size=hidden_size,num_layers=num_layers,output_size=output_size,feature_type=feature_type)
            
            if cf.use_boosting:
                self.group2 = CNN_LSTM_Group(input_size=input_size,hidden_size=hidden_size,num_layers=1,output_size=output_size,feature_type=feature_type)
                self.group3 = CNN_LSTM_Group(input_size=input_size,hidden_size=hidden_size,num_layers=num_layers,output_size=output_size,feature_type=feature_type)
                self.groups = [self.group, self.group2, self.group3]
                # self.groups = [self.group for _ in range(3)]

    def forward(self, batch_features, input_lengths):
        if cf.debug: print('\n=============model forward begin')
        """
        :param batch_features: batched acoustic features
        :return: the output of your model (e.g., log probability)
        """

        if self.feature_type == "quantized":

            batch_features = batch_features.unsqueeze(1) # [batch, 1, seq_len, feature_len]
            # old version when using conv
            x = self.conv1(batch_features) # [batch, 1, seq_len', feature_len']
            x = self.bn1(x)
            x = F.relu(x)

            # if debug: print(f'after 1 conv x shape: {x.shape}')
            x = x.squeeze(1) # [batch, 1, seq_len', feature_len']
            if cf.debug: print(f'after second conv2d: {x.shape}')
            x = nn.utils.rnn.pack_padded_sequence(x,input_lengths,batch_first=True,enforce_sorted=False)
            if not use_gru:
                x,_ = self.lstm(x)
            else:
                x,_ = self.gru(x)
            
            x,_ = nn.utils.rnn.pad_packed_sequence(x,batch_first=True,padding_value=0)
            current_length = x.size(1)
            if cf.debug: print(f'after lstm padding length:{current_length}')
            if current_length < self.output_size[0]:
                padding = torch.zeros((x.size(0),self.output_size[0] - current_length,x.size(2))).to(x.device)
                x = torch.cat([x,padding],dim=1)
            if cf.use_transformer:
                x = self.transformer_encoder(x)

            if cf.debug: print(f'x after lstm shape:{x.shape}')

            x = x.view(x.shape[0] * x.shape[1],-1)

            if cf.debug: print(f'x before fc shape:{x.shape}')
            
            # x = self.dropout(x)
            x = self.fc(x)

            x = x.view(-1,self.output_size[0],self.output_size[1])
            if cf.debug: print(f'model output shape: {x.shape}')
            x = F.log_softmax(x,dim=-1)
            if cf.debug: print('=============model forward over\n')
            return x
        else:
            # old version input channel 1
            # # old version without conv
            # # x = batch_features
            # # new version with conv
            # batch_features = batch_features.unsqueeze(1) # [batch, 1, seq_len, feature_len]
            # x = self.conv1(batch_features) # [batch, 2, seq_len, feature_len]
            # x = self.bn1(x)
            # x = F.relu(x)
            if not cf.use_boosting:
                x = self.group(batch_features,input_lengths)
            else:
                x = self.groups[0](batch_features,input_lengths)
                y = self.groups[1](batch_features,input_lengths)
                z = self.groups[2](batch_features,input_lengths)

                x = torch.cat([x,y,z],dim=-1)

            return x
