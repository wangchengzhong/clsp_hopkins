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
from sklearn.preprocessing import StandardScaler

def get_feature_vec_model(vec_model_path, train_mode = False, feature_file = None):
    if train_mode:
        features = pd.read_csv(feature_file,header=None).values.tolist()[1:]
        features = [[feature for feature in feature_list[0].split(' ')if feature] for feature_list in features]
        model = Word2Vec(features,vector_size=256,min_count=1,workers=5)
        model.save(vec_model_path)
    else:
        model = Word2Vec.load(vec_model_path)
    return model

class Phonemes:
    def __init__(self,scr_file):
        g2p = G2p()
        script = np.array(pd.read_csv(scr_file, header=None).values.tolist()[1:]).flatten().tolist()
        self.words = list(set([word for word in script]))
        self.word_phonemes = [g2p(word) for word in self.words]
        self.word_to_phonemes = {word: g2p(word) for word in self.words}
        self.phonemes_to_word = {''.join(g2p(word)): word for word in script}

        self.phonemes = list(set([phoneme for sublist in self.word_to_phonemes.values() for phoneme in sublist]))
        self.phonemes_to_int = {phoneme: i+1 for i, phoneme in enumerate(self.phonemes)}
        self.int_to_phonemes = {i+1: phoneme for i, phoneme in enumerate(self.phonemes)}

    def words_to_phonemes_int(self, words):
        phonemes = [self.word_to_phonemes[word] for word in words]
        return [[self.phonemes_to_int[phoneme] for phoneme in one_word_phoneme] for one_word_phoneme in phonemes]

class AsrDataset(Dataset):
    def __init__(self, scr_file, feature_file=None,
                 feature_label_file=None,
                 wav_scp='data/clsp.trnwav', wav_dir='data/waveforms',feature_type=cf.feature_type):
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
        self.phonemes = None
        self.script = np.array(pd.read_csv(scr_file, header=None).values.tolist()[1:]).flatten().tolist()
        # # new version when using g2p to convert script to phonemes
        if cf.use_phoneme:
            self.phoneme_class = Phonemes(scr_file)
            self.script = self.phoneme_class.words_to_phonemes_int(self.script)
            self.script = [[0] + array + [0] for array in self.script]
        else:
            # old version when script is not phoneme but word itself
            self.script = [[self.letter_to_int(c) for c in str] for str in self.script ]

            # old version when only using 0 at start and end of array
            
            self.script = [[0] + array + [0] for array in self.script]

            # new version when add 0 at each interval
            # self.script = [[0] + [item for sublist in [[i,0] for i in array] for item in sublist] for array in self.script]

        if feature_type == 'quantized':

            self.features = pd.read_csv(feature_file,header=None).values.tolist()[1:]
            
            if cf.use_vectorized_feature:
                # new version when features is converted by Word2Vec
                self.features = [[feature for feature in feature_list[0].split(' ')if feature] for feature_list in self.features]
                # self.model = Word2Vec(self.features,vector_size=256,min_count=1,workers=5)
                self.model = get_feature_vec_model(cf.word_vec_path, train_mode=False, feature_file=feature_file)
                # # print(f"model testing: {self.model.wv['GQ']}")
            else:
                # old version when features is one_hot
                self.labels = np.array(pd.read_csv(feature_label_file, header=None).values.tolist()[1:]).flatten().tolist()
                feature_to_index = {''.join(code): i for i, code in enumerate(self.labels)}
                self.features = [[feature_to_index[feature] for feature in feature_list[0].split(' ')if feature] for feature_list in self.features]
        else:
            self.features = self.compute_mfcc(wav_scp = wav_scp, wav_dir = wav_dir)
        
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
        if self.feature_type == 'quantized':
            if cf.use_vectorized_feature:
                # new version when feature is processed by Word2Vec
                feature = [self.model.wv[a] for a in self.features[idx]]
            else:
                # old version when feature is one-hot
                feature = self.features[idx] # older version in one-hot np.eye(256)[np.array(self.features[idx])-1]
        else: # self.feature: mfcc
            feature = self.features[idx]
        return feature, spelling_of_word

    def letter_to_int(self,char):
        return string.ascii_lowercase.index(char.lower())+1


    # This function is provided
    def compute_mfcc(self, wav_scp, wav_dir):
        """
        Compute MFCC acoustic features (dim=40) for each wav file.
        :param wav_scp:
        :param wav_dir:
        :return: features: List[np.ndarray, ...]
        """
        scaler = StandardScaler()
        features = []
        with open(wav_scp, 'r') as f:
            for wavfile in f:
                wavfile = wavfile.strip()
                if wavfile == 'jhucsp.trnwav' or wavfile == 'clsp.trnwav':  # skip header
                    continue
                wavfile_path = os.path.join(wav_dir, wavfile)
                # assert(os.path.isfile(wavfile_path),f"文件{wavfile_path}不存在")
                wav, sr = sf.read(os.path.join(wav_dir, wavfile))
                # wav = wav[np.nonzero(wav)[0]]
                feats = librosa.feature.mfcc(y=wav, sr=16e3, n_mfcc=40, hop_length=160, win_length=400).transpose()
                delta_mfccs = librosa.feature.delta(feats)
                delta2_mfccs = librosa.feature.delta(feats,order=2)
                feats = np.concatenate([feats, delta_mfccs, delta2_mfccs],axis=1)
                feats = scaler.fit_transform(feats)
                features.append(feats)
        return features
###########################test module###############################
# training_set = AsrDataset(scr_file='data/clsp.trnscr',feature_file='data/clsp.trnlbls',feature_label_file='data/clsp.lblnames',wav_scp='data/clsp.trnwav',wav_dir='data/waveforms')
# print(np.max([len(a) for a in training_set.features]))