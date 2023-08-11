import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.init as init
from config import debug,in_seq_length,hidden_size
class LSTM_ASR(torch.nn.Module):
    def __init__(self, input_size=[in_seq_length,256], hidden_size=hidden_size, num_layers=1,
                 output_size=[62,26],feature_type="quantized"):
        super().__init__()
        if debug: print('\n============model initializing start')
        self.output_size = output_size
        self.feature_type = feature_type
        
        # cancel conv2d
        if feature_type == "quantized":
            self.conv1 = nn.Conv2d(1, 1, kernel_size=3,stride=2)
            # self.conv2 = nn.Conv2d(1, 1, kernel_size=3,stride=2)
            self.bn1 = nn.BatchNorm2d(1)

            # when using mfcc
            self.into_lstm_seq_length = 127 # 19
        else:
            self.into_lstm_seq_length = 120
        self.lstm = nn.LSTM(input_size=self.into_lstm_seq_length,hidden_size=hidden_size,num_layers=num_layers,batch_first=True,bidirectional=True)
        # self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(in_features = hidden_size * 2, out_features = output_size[1])
        if debug: print('============model initializing finished\n')
        # for m in self.modules():
        #     if isinstance(m,nn.Conv2d):
        #         init.kaiming_normal_(m.weight)
        #         if m.bisa is not None:
        #             init.constant_(m.bias,0)
        #     elif isinstance(m, nn.BatchNorm2d):
        #         init.constant_(m.weight,1)
        #         init.constant_(m.bias,0)

    def forward(self, batch_features, input_lengths):
        if debug: print('\n=============model forward begin')
        """
        :param batch_features: batched acoustic features
        :return: the output of your model (e.g., log probability)
        """

        if self.feature_type == "quantized":

            # old version when using conv2d
            batch_features = batch_features.unsqueeze(1)
            # old version when using conv
            x = self.conv1(batch_features)
            x = self.bn1(x)
            x = F.relu(x)

            # if debug: print(f'after 1 conv x shape: {x.shape}')
            # x = F.relu(self.conv2(x))
            x = x.squeeze(1)

        else:
            # new version without conv
            x = batch_features

        if debug: print(f'after second conv2d: {x.shape}')
        x = nn.utils.rnn.pack_padded_sequence(x,input_lengths,batch_first=True,enforce_sorted=False)
        x,_ = self.lstm(x)

        x,_ = nn.utils.rnn.pad_packed_sequence(x,batch_first=True,padding_value=0)
        current_length = x.size(1)
        if current_length < self.output_size[0]:
            padding = torch.zeros((x.size(0),self.output_size[0] - current_length,x.size(2))).to(x.device)
            x = torch.cat([x,padding],dim=1)
        # new version when directly output lstm
        # x = x.log_softmax(-1)
        
        # old version when using fc
        # x = x.transpose(0,1)
        if debug: print(f'x after lstm shape:{x.shape}')
        # old version when linear layer is processed throughout all sequence_len*feautre_len
        # x = x.reshape(x.size(0),-1)
        # new version when linear layer is only processed through feature_len
        x = x.view(x.shape[0] * x.shape[1],-1)

        if debug: print(f'x before fc shape:{x.shape}')
        
        # x = self.dropout(x)
        x = self.fc(x)

        x = x.view(-1,self.output_size[0],self.output_size[1])
        if debug: print(f'model output shape: {x.shape}')
        x = F.log_softmax(x,dim=-1)
        if debug: print('=============model forward over\n')
        return x