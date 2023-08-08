import string
import torch
import librosa
import os
import pandas as pd
from torch.utils.data import Dataset
import numpy as np
from gensim.models import Word2Vec
from g2p_en import G2p
from collections import Counter
import soundfile as sf
import config as cf
class AsrDataset(Dataset):
    def __init__(self, scr_file, feature_file=None,
                 feature_label_file=None,
                 wav_scp='data/clsp.trnwav', wav_dir='data/waveforms',feature_type='quantized'):
        """
        :param scr_file: clsp.trnscr
        :param feature_type: "quantized" or "mfcc"
        :param feature_file: clsp.trainlbls or clsp.devlbls
        :param feature_label_file: clsp.lblnames
        :param wav_scp: clsp.trnwav or clsp.devwav
        :param wav_dir: wavforms/
        """
        g2p = G2p()
        self.feature_type = feature_type
        assert self.feature_type in ['quantized', 'mfcc']

        self.blank = "<blank>"
        self.silence = "{"

        # load data
        self.script = np.array(pd.read_csv(scr_file, header=None).values.tolist()[1:]).flatten().tolist()
        # # new version when using g2p to convert script to phonemes
        if cf.use_phoneme:
            self.script = [g2p(word) for word in self.script]
            self.phonemes = list(set([phoneme for sublist in self.script for phoneme in sublist]))
            # self.phonemes_counts = Counter([phoneme for sublist in self.script for phoneme in sublist])
            phonemes_to_int = {phoneme: i+1 for i, phoneme in enumerate(self.phonemes)}
            self.script = [[phonemes_to_int[phoneme] for phoneme in one_word_phoneme]for one_word_phoneme in self.script]
            self.script = [[0] + array + [0] for array in self.script] # max length (including 0): 10
        else:
            # old version when script is not phoneme but word itself
            self.script = [[self.char_to_int(c) for c in str] for str in self.script ]

            # old version when only using 0 at start and end of array
            self.script = [[0] + array + [0] for array in self.script]

            # new version when add 0 at each interval
            # self.script = [[0] + [item for sublist in [[i,0] for i in array] for item in sublist] for array in self.script]

        
        self.features = pd.read_csv(feature_file,header=None).values.tolist()[1:]
        
        if cf.use_vectorized_feature:
            # new version when features is converted by Word2Vec
            self.features = [[feature for feature in feature_list[0].split(' ')if feature] for feature_list in self.features]
            self.model = Word2Vec(self.features,vector_size=256,min_count=1,workers=5)
            # # print(f"model testing: {self.model.wv['GQ']}")
        else:
            # old version when features is one_hot
            self.labels = np.array(pd.read_csv(feature_label_file, header=None).values.tolist()[1:]).flatten().tolist()
            code_to_index = {''.join(code): i for i, code in enumerate(self.labels)}
            self.features = [[code_to_index[feature] for feature in feature_list[0].split(' ')if feature] for feature_list in self.features]

        # self.mfcc_features = self.compute_mfcc(wav_scp = wav_scp, wav_dir = wav_dir)
        
        self.max_feature_length = np.max([len(a) for a in self.features])
        self.max_scipt_length = np.max([len(a) for a in self.script])

        # tmp = np.array(pd.read_csv('data/clsp.endpts',header=None).values.tolist()[1:]).flatten().tolist()
        # self.endpoints = [[int(start),int(end)] for start_end_str in tmp for start,end in [start_end_str.split(' ')] ]
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
        
        spelling_of_word = self.script[idx]# older version in one-hot np.eye(26)[np.array(self.script[idx])-1]
        if cf.use_vectorized_feature:
            # new version when feature is processed by Word2Vec
            feature = [self.model.wv[a] for a in self.features[idx]]
        else:
            # old version when feature is one-hot
            feature = self.features[idx] # older version in one-hot np.eye(256)[np.array(self.features[idx])-1]

        return feature, spelling_of_word

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
                wavfile_path = os.path.join(wav_dir, wavfile)
                # assert(os.path.isfile(wavfile_path),f"文件{wavfile_path}不存在")
                wav, sr = sf.read(os.path.join(wav_dir, wavfile),dtype='float32')
                feats = librosa.feature.mfcc(y=wav, sr=16e3, n_mfcc=40, hop_length=160, win_length=400).transpose()
                features.append(feats)
        return features
###########################test module###############################
# training_set = AsrDataset(scr_file='data/clsp.trnscr',feature_file='data/clsp.trnlbls',feature_label_file='data/clsp.lblnames')
# print(training_set.script)