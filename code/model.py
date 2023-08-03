import torch
import torch.nn as nn

class LSTM_ASR(torch.nn.Module):
    def __init__(self, input_size=256, hidden_size=512, num_layers=2,
                 output_size=26,feature_type="quantized",batch_first = False):
        super().__init__()

        assert feature_type in ['quantized', 'mfcc']
        self.lstm = nn.LSTM(input_size,hidden_size,num_layers,batch_first = False)

        self.dropout = nn.Dropout(0.5)
        
        self.fc1 = nn.Linear(hidden_size,64)
        self.fc2 = nn.Linear(64,output_size)


    def forward(self, batch_features):
        """
        :param batch_features: batched acoustic features
        :return: the output of your model (e.g., log probability)
        """

        lstm_out,_ = self.lstm(batch_features)
        out = self.dropout(lstm_out)
        out = self.fc1(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out