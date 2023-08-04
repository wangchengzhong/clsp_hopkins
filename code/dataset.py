import string
import torch
import librosa
import os
import pandas as pd
from torch.utils.data import Dataset
import numpy as np

class AsrDataset(Dataset):
    def __init__(self, scr_file, feature_file=None,
                 feature_label_file=None,
                 wav_scp=None, wav_dir=None,feature_type='quantized'):
        """
        :param scr_file: clsp.trnscr
        :param feature_type: "quantized" or "mfcc"
        :param feature_file: clsp.trainlbls or clsp.devlbls
        :param feature_label_file: clsp.lblnames
        :param wav_scp: clsp.trnwav or clsp.devwav
        :param wav_dir: wavforms/
        """
        self.feature_type = feature_type
        assert self.feature_type in ['quantized', 'mfcc']

        self.blank = "<blank>"
        self.silence = "{"

        # load data
        self.script = np.array(pd.read_csv(scr_file, header=None).values.tolist()[1:]).flatten().tolist()

        self.script = [[self.char_to_int(c) for c in str] for str in self.script ]
        self.script = [[0]+ array + [0] for array in self.script]

        self.features = pd.read_csv(feature_file,header=None).values.tolist()[1:]
        
        self.labels = np.array(pd.read_csv(feature_label_file, header=None).values.tolist()[1:]).flatten().tolist()
        
        code_to_index = {''.join(code): i for i, code in enumerate(self.labels)}

        tmp = np.array(pd.read_csv('data/clsp.endpts',header=None).values.tolist()[1:]).flatten().tolist()
        self.endpoints = [[int(start),int(end)] for start_end_str in tmp for start,end in [start_end_str.split(' ')] ]
        # index_to_code = {i:code for i,code in enumerate(codes)}

        self.features = [[code_to_index[feature] for feature in feature_list[0].split(' ')if feature] for feature_list in self.features]
        self.max_feature_length = np.max([len(a) for a in self.features])
        self.max_scipt_length = np.max([len(a) for a in self.script])


        # print(self.features[0][0])
    def __len__(self):
        """
        :return: num_of_samples
        """
        return len(self.script)

    def __getitem__(self, idx):
        """
        Get one sample each time. Do not forget the leading- and trailing-silence.
        :param idx: index of sample
        :return: spelling_of_word, feature
        """
        
        spelling_of_word = self.script[idx]# np.eye(26)[np.array(self.script[idx])-1]
        feature = self.features[idx]# np.eye(256)[np.array(self.features[idx])-1]

        # endpoint = self.endpoints[idx]
        return feature,spelling_of_word

    def char_to_int(self,char):
        return string.ascii_lowercase.index(char.lower())+1


    # This function is provided
    def compute_mfcc(self, wav_scp, wav_dir):
        """
        Compute MFCC acoustic features (dim=40) for each wav file.
        :param wav_scp:
        :param wav_dir:
        :return: features: List[np.ndarray, ...]
        """
        features = []
        with open(wav_scp, 'r') as f:
            for wavfile in f:
                wavfile = wavfile.strip()
                if wavfile == 'jhucsp.trnwav':  # skip header
                    continue
                wav, sr = librosa.load(os.path.join(wav_dir, wavfile), sr=None)
                feats = librosa.feature.mfcc(y=wav, sr=16e3, n_mfcc=40, hop_length=160, win_length=400).transpose()
                features.append(feats)
        return features
###########################test module###############################
# training_set = AsrDataset('data/clsp.trnscr','data/clsp.trnlbls','data/clsp.lblnames')
# print(training_set.max_feature_length)