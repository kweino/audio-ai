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


# # Load the environment variables from the .env file in the parent folder
# dotenv_path = os.path.join(os.path.dirname(__file__), '../', '.env')
# load_dotenv(dotenv_path)
# # Set the environment variable using os.getenv()
# ANNOTATIONS_FILE = os.getenv('ANNOTATIONS_FILE')
# AUDIO_DIR = os.getenv('AUDIO_DIR')

SAMPLE_RATE = 22050
NUM_SAMPLES = 22050
DURATION = 5 # sec(s)

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

device = device
transformation = mel_spectrogram.to(device)
target_sample_rate = SAMPLE_RATE
target_num_samples = NUM_SAMPLES * DURATION


def _resample_if_necessary(signal, sr):
    if sr != target_sample_rate:
        resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
        signal = resampler(signal)
    return signal

def _mix_down_if_necessary(signal):
    '''signal -> (num_channels, samples). Transforms (2, 16000) -> (1, 16000)'''
    if signal.shape[0] > 1:
        signal = torch.mean(signal, dim=0, keepdim=True)
    return signal

def _cut_if_necessary(signal):
    # signal -> Tensor(1, num_samples)
    if signal.shape[1] > target_num_samples:
        signal = signal[:,:target_num_samples]
    return signal

def _right_pad_if_necessary(signal):
    signal_length = signal.shape[1]
    if signal_length < target_num_samples:
        num_missing_samples = target_num_samples - signal_length
        last_dim_padding = (0,num_missing_samples) # (right pad, left pad)
        signal = torch.nn.functional.pad(signal,last_dim_padding)
    return signal

def load_new_sound():
    pass

def _capture_mic_input(duration, num_samples, sr):
    # Capture audio from the microphone
    print("Recording audio...")
    recording = sd.rec(duration * num_samples, sr, channels=1)
    sd.wait()

    print(f"Recording finished. \n Raw array shape: {recording.shape}")
    return recording

def _preprocess_audio(recording):
    print('Processing audio...')
    # Convert the audio data to a tensor
    signal = torch.from_numpy(np.asarray(recording)).view(1,-1).float()
    print(f'''signal tensor type: {type(signal)} \n shape: {signal.shape}''')
    sr = target_sample_rate

    # Preprocess & return  signal
    signal = signal.to(device) #register signal on the correct hardware (CPU or GPU)
    signal = _resample_if_necessary(signal, sr) # standardize sample rate
    signal = _mix_down_if_necessary(signal) # mix down to mono
    signal = _right_pad_if_necessary(signal) # zero pad undersampled audio
    signal = _cut_if_necessary(signal) # trim sounds longer than target sr
    signal = transformation(signal) # apply passed-in transformation
    signal =  signal.numpy()
    return signal 

def plot_spectrogram(mel_spec, title=None):
    print('Plotting.')
    # import matplotlib
    # matplotlib.use('svg')

    # mel_spec = mel_spec.numpy()
    S_db = librosa.power_to_db(mel_spec**2,ref=np.max) 
    print('power to db done.')
    # librosa.display.specshow(ref=np.max), 
    #                          y_axis='mel', fmax=8000, x_axis='time')
    fig, ax = plt.subplots()
    print('figure created')
    img = librosa.display.specshow(S_db, ax=ax, y_axis='mel', fmax=8000, x_axis='time')
    print('img var created')
    fig.colorbar(img, ax=ax)
    # plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-frequency spectrogram')
    fig.tight_layout()
    plt.show(block=True)
    # plt.savefig('mel.svg')

if __name__ == '__main__':
   
    recording = _capture_mic_input(DURATION, NUM_SAMPLES, SAMPLE_RATE)
    print(f'''raw recording array type: {type(recording)} \n shape: {recording.shape}''')
    signal = _preprocess_audio(recording)

    print(f'''mel-spec array type: {type(signal)} \n shape: {signal.shape}''')
    print(f'''mel-spec array type: {type(signal[0])} \n shape: {signal[0].shape}''')
    # # Display the melspectrogram
    plot_spectrogram(signal[0])
    print('Plot complete.')
    
