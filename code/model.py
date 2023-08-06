import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
global debug
debug = False
class LSTM_ASR(torch.nn.Module):
    def __init__(self, input_size=[190,256], hidden_size=3, num_layers=1,
                 output_size=[190,26],feature_type="quantized"):
        super().__init__()

        self.output_size = output_size
        assert feature_type in ['quantized', 'mfcc']

        # self.conv1 = nn.Conv2d(1, 1, kernel_size=3,stride=2)
        # self.conv2 = nn.Conv2d(1, 1, kernel_size=3,stride=2)
        
        conv_out_seq_length = input_size[1]
        self.lstm = nn.LSTM(input_size=conv_out_seq_length,hidden_size=hidden_size,num_layers=num_layers,batch_first=True,bidirectional=True)
        self.fc = nn.Linear(in_features = hidden_size * 2, out_features = output_size[1])

    def forward(self, batch_features, input_lengths):
        """
        :param batch_features: batched acoustic features
        :return: the output of your model (e.g., log probability)
        """
        #old version when using conv
        # x = F.relu(self.conv1(batch_features))

        # if debug: print(f'after 1 conv x shape: {x.shape}')
        # x = F.relu(self.conv2(x))
        # x = x.squeeze(1)

        # new version without conv
        x = batch_features
        if debug: print(f'after second conv2d: {x.shape}')
        x = nn.utils.rnn.pack_padded_sequence(x,input_lengths,batch_first=True,enforce_sorted=False)
        x,_ = self.lstm(x)

        x,_ = nn.utils.rnn.pad_packed_sequence(x,batch_first=True,padding_value=0)
        current_length = x.size(1)
        if current_length < 190:
            padding = torch.zeros((x.size(0),190-current_length,x.size(2))).to(x.device)
            x = torch.cat([x,padding],dim=1)
        # new version when directly output lstm
        # x = x.log_softmax(-1)
        
        # old version when using fc
        # x = x.transpose(0,1)
        if debug: print(f'x after lstm shape:{x.shape}')
        # old version when linear layer is processed throughout all sequence_len*feautre_len
        # x = x.reshape(x.size(0),-1)
        # new version when linear layer is only processed through feature_len
        x = x.reshape(x.shape[0]*x.shape[1],-1)

        if debug: print(f'x before fc shape:{x.shape}')
        
        x = self.fc(x)
        x = x.reshape(-1,self.output_size[0],self.output_size[1])
        if debug: print(f'model output shape: {x.shape}')
        return x