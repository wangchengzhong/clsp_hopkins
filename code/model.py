import torch


class LSTM_ASR(torch.nn.Module):
    def __init__(self, feature_type="discrete", input_size=64, hidden_size=256, num_layers=2,
                 output_size=28):
        super().__init__()
        assert feature_type in ['discrete', 'mfcc']
        # Build your own neural network. Play with different hyper-parameters and architectures.
        # === write your code here ===
        pass



    def forward(self, batch_features):
        """
        :param batch_features: batched acoustic features
        :return: the output of your model (e.g., log probability)
        """
        # === write your code here ===
        pass
