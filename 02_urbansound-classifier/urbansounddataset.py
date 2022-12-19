import os
from dotenv import load_dotenv

import torch
from torch.utils.data import Dataset
import pandas as pd
import torchaudio


class UrbanSoundDataset(Dataset):

    def __init__(self, 
                 annotations_file, 
                 audio_dir, 
                 transformation, 
                 target_sample_rate,
                 num_samples,
                 device):
        self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir
        self.device = device
        self.transformation = transformation.to(self.device)
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples
        

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, sr = torchaudio.load(audio_sample_path) 
        signal = signal.to(self.device) #register signal on the correct hardware (CPU or GPU)
        signal = self._resample_if_necessary(signal, sr) # standardize sample rate
        signal = self._mix_down_if_necessary(signal) # mix down to mono
        signal = self._right_pad_if_necessary(signal) # zero pad undersampled audio
        signal = self._cut_if_necessary(signal) # trim sounds longer than target sr
        signal = self.transformation(signal) # apply passed-in transformation 
        return signal, label
    
    

    def _get_audio_sample_path(self, index):
        fold = f'fold{self.annotations.iloc[index, 5]}'
        path = os.path.join(self.audio_dir, fold, self.annotations.iloc[index, 0])
        return path
    
    def _get_audio_sample_label(self, index):
        return self.annotations.iloc[index, 6]
    
    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        return signal

    def _mix_down_if_necessary(self, signal):
        '''signal -> (num_channels, samples). Transforms (2, 16000) -> (1, 16000)'''
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _cut_if_necessary(self,signal):
        # signal -> Tensor(1, num_samples)
        if signal.shape[1] > self.num_samples:
            signal = signal[:,:self.num_samples]
        return signal

    def _right_pad_if_necessary(self, signal):
        signal_length = signal.shape[1]
        if signal_length < self.num_samples:
            num_missing_samples = self.num_samples - signal_length
            last_dim_padding = (0,num_missing_samples) # (right pad, left pad)
            signal = torch.nn.functional.pad(signal,last_dim_padding)
        return signal

if __name__ == '__main__':
    # Load the environment variables from the .env file in the parent folder
    dotenv_path = os.path.join(os.path.dirname(__file__), '../', '.env')
    load_dotenv(dotenv_path)
    # Set the environment variable using os.getenv()
    ANNOTATIONS_FILE = os.getenv('ANNOTATIONS_FILE')
    AUDIO_DIR = os.getenv('AUDIO_DIR')

    SAMPLE_RATE = 22050
    NUM_SAMPLES = 22050

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print(device)

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    usd = UrbanSoundDataset(ANNOTATIONS_FILE, 
                            AUDIO_DIR, 
                            mel_spectrogram, 
                            SAMPLE_RATE,
                            NUM_SAMPLES,
                            device)

    print(f'There are {len(usd)} samples in the dataset')

    signal, label  = usd[0]
    print(signal.shape, label)