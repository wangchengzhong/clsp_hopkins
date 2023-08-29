import string
import pandas as pd
from torch.utils.data import Dataset
import numpy as np
import config as cf
if cf.use_vectorized_feature:
    from gensim.models import Word2Vec

def get_feature_vec_model(vec_model_path, train_mode = False, feature_file = None):
    if train_mode:
        features = pd.read_csv(feature_file,header=None).values.tolist()[1:]
        features = [[feature for feature in feature_list[0].split(' ')if feature] for feature_list in features]
        model = Word2Vec(features,vector_size=256,min_count=1,workers=5)
        model.save(vec_model_path)
    else:
        model = Word2Vec.load(vec_model_path)
    return model

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
        # old version when script is not phoneme but word itself
        self.script = [[self.letter_to_int(c) for c in str] for str in self.script ]
     
        self.script = [[0] + array + [0] for array in self.script]

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
        return feature, spelling_of_word

    def letter_to_int(self,char):
        return string.ascii_lowercase.index(char.lower())+1



###########################test module###############################
# training_set = AsrDataset(scr_file='data/split/clsp.trnscr.kept',feature_file='data/split/clsp.trnlbls.kept',feature_label_file='data/clsp.lblnames',wav_scp='data/clsp.trnwav',wav_dir='data/waveforms')
# print(np.max([len(a) for a in training_set.features]))

    

