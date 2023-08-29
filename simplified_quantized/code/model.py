import torch
import torch.nn as nn
import torch.nn.functional as F
import config as cf

class LSTM_ASR(torch.nn.Module):
    def __init__(self, input_size=[cf.in_seq_length,256], hidden_size=cf.hidden_size, num_layers=cf.num_layers,
                 output_size=[62,26],feature_type="quantized"):
        super().__init__()
        self.output_size = output_size
        self.feature_type = feature_type

        if feature_type == "quantized":
            self.conv1 = nn.Conv2d(1, 1, kernel_size=3,stride=2)
            self.bn1 = nn.BatchNorm2d(1)

            self.into_lstm_seq_length = 127 # 19

            self.lstm = nn.LSTM(input_size=self.into_lstm_seq_length,hidden_size=hidden_size,num_layers=num_layers,batch_first=True,bidirectional=True)
            # self.dropout = nn.Dropout(0.5)
            self.fc = nn.Linear(in_features = hidden_size * 2, out_features = output_size[1])

    def forward(self, batch_features, input_lengths):
        """
        :param batch_features: batched acoustic features
        :return: the output of your model (e.g., log probability)
        """

        if self.feature_type == "quantized":

            batch_features = batch_features.unsqueeze(1) # [batch, 1, seq_len, feature_len]

            x = self.conv1(batch_features) # [batch, 1, seq_len', feature_len']
            x = self.bn1(x)
            x = F.relu(x)
            x = x.squeeze(1) # [batch, 1, seq_len', feature_len']
            if cf.debug: print(f'after second conv2d: {x.shape}')
            x = nn.utils.rnn.pack_padded_sequence(x,input_lengths,batch_first=True,enforce_sorted=False)

            x,_ = self.lstm(x)

            x,_ = nn.utils.rnn.pad_packed_sequence(x,batch_first=True,padding_value=0)
            current_length = x.size(1)
            if cf.debug: print(f'after lstm padding length:{current_length}')
            if current_length < self.output_size[0]:
                padding = torch.zeros((x.size(0),self.output_size[0] - current_length,x.size(2))).to(x.device)
                x = torch.cat([x,padding],dim=1)
            x = x.view(x.shape[0] * x.shape[1],-1)   
            # x = self.dropout(x)
            x = self.fc(x)

            x = x.view(-1,self.output_size[0],self.output_size[1])
            if cf.debug: print(f'model output shape: {x.shape}')
            x = F.log_softmax(x,dim=-1)
            if cf.debug: print('=============model forward over\n')
            return x

