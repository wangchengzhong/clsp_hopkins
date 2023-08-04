import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
global debug
debug = False
class LSTM_ASR(torch.nn.Module):
    def __init__(self, input_size=[256,256], hidden_size=26, num_layers=3,
                 output_size=[16,26],feature_type="quantized"):
        super().__init__()

        assert feature_type in ['quantized', 'mfcc']

        self.conv1 = nn.Conv2d(1,8,kernel_size=5,stride=3)
        self.conv2 = nn.Conv2d(8,1,kernel_size=5,stride=3)

        self.lstm = nn.LSTM(input_size=27,hidden_size=26,num_layers=2,batch_first=False)
        self.fc = nn.Linear(in_features = 27*26,out_features = 16*26)


    def forward(self, batch_features):
        """
        :param batch_features: batched acoustic features
        :return: the output of your model (e.g., log probability)
        """
        x = F.relu(self.conv1(batch_features))
        if debug: print(f'after 1 conv x shape: {x.shape}')
        x = F.relu(self.conv2(x))

        x = x.view(27,-1,27)
        # print(x.shape)
        x,_ = self.lstm(x)
        x = x.transpose(0,1)
        if debug: print(f'x after lstm shape:{x.shape}')
        x = x.reshape(x.size(0),-1)
        # print(x.shape)
        x = self.fc(x)
        x = x.view(-1,16,26)
        return x