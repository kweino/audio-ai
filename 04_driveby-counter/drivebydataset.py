import os
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sounddevice as sd
import torch
from torch.utils.data import Dataset
import torchaudio
import librosa
import librosa.display




class DriveByDataset(Dataset):

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
        # audio_sample_path = self._get_audio_sample_path(index)
        # label = self._get_audio_sample_label(index)
        # signal, sr = torchaudio.load(audio_sample_path) 
        # signal = signal.to(self.device) #register signal on the correct hardware (CPU or GPU)
        # signal = self._resample_if_necessary(signal, sr) # standardize sample rate
        # signal = self._mix_down_if_necessary(signal) # mix down to mono
        # signal = self._right_pad_if_necessary(signal) # zero pad undersampled audio
        # signal = self._cut_if_necessary(signal) # trim sounds longer than target sr
        # signal = self.transformation(signal) # apply passed-in transformation 
        return signal#, label
    

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

    def load_new_sound(self):
        pass

    def _capture_mic_input(self, duration):
        sr = self.target_sample_rate
        num_samples = self.num_samples

        # Capture audio from the microphone
        print("Recording audio...")
        recording = sd.rec(duration * num_samples, sr, channels=1)
        sd.wait()

        print(f"Recording finished. \n Raw array shape: {recording.shape}")
        return recording
    
    def _preprocess_audio(self,recording):
        print('Processing audio...')
        # Convert the audio data to a tensor
        signal = torch.from_numpy(np.asarray(recording)).view(1,-1).float()
        print(f'''signal tensor type: {type(signal)} \n shape: {signal.shape}''')
        sr = self.target_sample_rate

        # Preprocess & return  signal
        signal = signal.to(self.device) #register signal on the correct hardware (CPU or GPU)
        signal = self._resample_if_necessary(signal, sr) # standardize sample rate
        signal = self._mix_down_if_necessary(signal) # mix down to mono
        signal = self._right_pad_if_necessary(signal) # zero pad undersampled audio
        signal = self._cut_if_necessary(signal) # trim sounds longer than target sr
        signal = self.transformation(signal) # apply passed-in transformation
        signal =  signal.numpy()
        return signal 

    def plot_spectrogram(mel_spec, title=None):
        print('Plotting.')
        # import matplotlib
        # matplotlib.use('svg')

        # mel_spec = mel_spec.numpy()
        S_db = librosa.power_to_db(mel_spec**2,ref=np.max) 
        print('power to db done.')
        # librosa.display.specshow(, ref=np.max), 
        #                          y_axis='mel', fmax=8000, x_axis='time')
        fig, ax = plt.subplots()
        print('figure created')
        img = librosa.display.specshow(S_db, ax=ax, y_axis='mel', fmax=8000, x_axis='time')
        print('img var created')
        fig.colorbar(img, ax=ax)
        # plt.colorbar(format='%+2.0f dB')
        fig.title('Mel-frequency spectrogram')
        fig.tight_layout()
        # plt.show(block=True)
        plt.savefig('mel.svg')

if __name__ == '__main__':
    import time

    t1 = time.time()

    # Load the environment variables from the .env file in the parent folder
    dotenv_path = os.path.join(os.path.dirname(__file__), '../', '.env')
    load_dotenv(dotenv_path)
    # Set the environment variable using os.getenv()
    ANNOTATIONS_FILE = os.getenv('ANNOTATIONS_FILE')
    AUDIO_DIR = os.getenv('AUDIO_DIR')

    SAMPLE_RATE = 22050
    NUM_SAMPLES = 22050
    DURATION = 1 # sec(s)

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print(f'Using {device} for processing.')

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    dbd = DriveByDataset(ANNOTATIONS_FILE, 
                            AUDIO_DIR, 
                            mel_spectrogram, 
                            SAMPLE_RATE,
                            NUM_SAMPLES,
                            device)

    
    recording = dbd._capture_mic_input(DURATION)
    print(f'''raw recording array type: {type(recording)} \n shape: {recording.shape}''')
    signal = dbd._preprocess_audio(recording)

    print(f'''mel-spec array type: {type(signal)} \n shape: {signal.shape}''')
    print(f'''mel-spec array type: {type(signal[0])} \n shape: {signal[0].shape}''')
    # # Display the melspectrogram
    dbd.plot_spectrogram(signal[0])
    print('Plot complete.')
    t2=time.time()
    print(f'Script executed in {t2-t1}s')
