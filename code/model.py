import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
global debug
debug = False
class LSTM_ASR(torch.nn.Module):
    def __init__(self, input_size=[190,256], hidden_size=26, num_layers=3,
                 output_size=[190,43],feature_type="quantized"):
        super().__init__()
        self.output_size = output_size
        assert feature_type in ['quantized', 'mfcc']

        self.conv1 = nn.Conv1d(190,190,7,4)
        # self.conv1 = nn.Conv2d(1,1,kernel_size=5,stride=3)
        # self.conv2 = nn.Conv2d(1,1,kernel_size=5,stride=3)

        self.lstm = nn.LSTM(input_size=63,hidden_size=output_size[1],num_layers=2,batch_first=True)
        # self.fc = nn.Linear(in_features = 127*output_size[1],out_features = output_size[0]*output_size[1])


    def forward(self, batch_features):
        """
        :param batch_features: batched acoustic features
        :return: the output of your model (e.g., log probability)
        """
        # batch_features = batch_features.transpose(1,2)

        x = F.relu(self.conv1(batch_features))
        # x = batch_features
        if debug: print(f'after 1 conv x shape: {x.shape}')
       
        # x = F.relu(self.conv2(x))

        # x = x.view(27,-1,27)
        # print(x.shape)
        x,_ = self.lstm(x)

        if debug: print(f'x after lstm shape:{x.shape}')
        x = x.reshape(x.size(0),-1)
        # print(x.shape)
        # x = self.fc(x)
        x = x.view(-1,self.output_size[0],self.output_size[1])
        x = x.log_softmax(-1)
        return x