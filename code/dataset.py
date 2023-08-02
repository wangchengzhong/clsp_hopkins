import string
import torch
import librosa
import os
import pandas as pd
from torch.utils.data import Dataset

class AsrDataset(Dataset):
    def __init__(self, scr_file, feature_type='discrete', feature_file=None,
                 feature_label_file=None,
                 wav_scp=None, wav_dir=None):
        """
        :param scr_file: clsp.trnscr
        :param feature_type: "quantized" or "mfcc"
        :param feature_file: clsp.trainlbls or clsp.devlbls
        :param feature_label_file: clsp.lblnames
        :param wav_scp: clsp.trnwav or clsp.devwav
        :param wav_dir: wavforms/
        """
        assert self.feature_type in ['discrete', 'mfcc']

        self.blank = "<blank>"
        self.silence = "<sil>"

        # === write your code here ===


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
        # === write your code here ===
        pass


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
