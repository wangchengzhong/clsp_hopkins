import os 
import shutil
from scipy.signal import butter, iirfilter, lfilter
import soundfile as sf

#############################DATA EXTENSION#############################
class DataExtension:
    def __init__(self):
        pass
    def butter_filter(self,cutoff,fs=16000,order=5,mode='lowpass',cutoff2 = None):
        # assert mode in ['lowpass','highpass','bandpass']
        nyq = 0.5*fs
        normal_cutoff = cutoff / nyq
        if mode == 'bandpass':
            high = cutoff2 / nyq
            b,a = butter(order, [normal_cutoff, high], btype=mode)
        else:
            b,a = butter(order, normal_cutoff, btype=mode,analog=False)
        return b,a
    def apply_shelving_iirfilter(self,data,cutoff,gain,fs,order=5,type='lowpass'):
        nyq = 0.5*16000
        normal_cutoff = cutoff / nyq
        b,a = iirfilter(order,normal_cutoff,btype=type,ftype='butter',output='ba')
        b *= 10**(gain/20)
        y = lfilter(b,a,data)
        return y
    def apply_low_high_band_butterfilter(self,data,filter_type,cutoff,order=5,fs=16000,cutoff2=None):
        b,a = self.butter_filter(cutoff=cutoff,mode=filter_type,fs=fs,cutoff2=cutoff2,order=order)
        y = lfilter(b,a,data)
        return y
    def write_filtered_file(self,data,filename,output_dir,filter_type,cutoff,cutoff2=None,sr=16000,gain=None):
        if not gain:
            new_data = self.apply_low_high_band_butterfilter(data=data,fs=sr,filter_type=filter_type,cutoff=cutoff)
            filename = filename.split('.wav')[0]
            new_filename = f'{filename}_{filter_type}_{cutoff}Hz.wav'
        else:
            new_data = self.apply_shelving_iirfilter(data=data,cutoff=cutoff,gain=gain,fs=sr,order=5,type=filter_type)
            filename = filename.split('.wav')[0]
            new_filename = f'{filename}_{filter_type}_{cutoff}Hz_{gain}db.wav'      
        new_filepath = os.path.join(output_dir,new_filename)
        sf.write(new_filepath,new_data,sr)
    def process_audio_files(self,wav_dir, output_dir):
        for filename in os.listdir(wav_dir):
            if filename.endswith(".wav"):
                filepath = os.path.join(wav_dir,filename)
                data,sr = sf.read(filepath)
                for cutoff in [3500,4000,4500,5000]:
                    self.write_filtered_file(data=data,filename=filename,output_dir=output_dir,filter_type='lowpass',cutoff=cutoff)
                for cutoff in [150,250,350,450]:
                    self.write_filtered_file(data=data,filename=filename,output_dir=output_dir,filter_type='highpass',cutoff=cutoff)
                for low_gain in [-5,-3,5,7,9]:
                    for low_freq in [500,1000,1500]:
                        self.write_filtered_file(data,filename,output_dir,filter_type='highpass',cutoff=int(low_freq),gain=low_gain)
                for high_gain in [-5,-3,5,7,9]:
                    for high_freq in [2500,3500,4500]:
                        self.write_filtered_file(data,filename,output_dir,filter_type='lowpass',cutoff=high_freq,gain=high_gain)


def copy_train_wav():
    with open('data/split/clsp.trnwav.kept','r') as f:
        wav_files = f.read().splitlines()[1:]
    os.makedirs('data/waveforms/1a',exist_ok=True)

    for wav_file in wav_files:
        source_file = os.path.join('data/waveforms',wav_file)
        target_file = os.path.join('data/waveforms/1a',wav_file)
        shutil.copy(source_file,target_file)

def make_clsp_trnwav_kept_extend_file():
    filenames = os.listdir('data/data_extend')
    with open('data/split/clsp.trnwav.kept.extend','w') as f:
        f.write('\n')
        for filename in filenames:
            f.write(filename+'\n')

def extend_clsp_trnscr_kept_file():
    with open('data/split/clsp.trnscr.kept','r') as f:
        lines = f.readlines()
    with open('data/split/clsp.trnscr.kept.extend','w') as f:
        for line in lines:
            if lines.index(line) == 0:
                f.write(line)
            else:
                f.write(line*38)
