import torch
import torchaudio
from cnn import CNNNEtwork
from train import  AUDIO_DIR, ANNOTATIONS_FILE, NUM_SAMPLES, SAMPLE_RATE
from urbansounddataset import UrbanSoundDataset


class_mapping = [
    'air_conditioner',
    'car_horn',
    'children_playing',
    'dog_bark',
    'drilling',
    'engine_idling',
    'gun_shot',
    'jackhammer',
    'siren',
    'street_music'
]

def predict(model, input, target, class_mapping):
    model.eval()
    with torch.no_grad():
        predictions = model(input)
        # Tensor (1,10) -> (num_inputs, num_classes)
        predicted_index = predictions[0].argmax(0)
        predicted = class_mapping[predicted_index]
        expected = class_mapping[target]
    return predicted, expected

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    # load model
    cnn = CNNNEtwork()
    state_dict = torch.load('cnnnet.pth')
    cnn.load_state_dict(state_dict)

    # instantiate mel spectrogram transformation
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    # instantiate urban sound dataset
    usd = UrbanSoundDataset(ANNOTATIONS_FILE, 
                            AUDIO_DIR, 
                            mel_spectrogram, 
                            SAMPLE_RATE,
                            NUM_SAMPLES,
                            device)

    # get sample from urban sound dataset for inference
    input, target = usd[1][0], usd[1][1] # [num_channels, fr, time]
    input.unsqueeze_(0) # [batch_size, num_channels, fr, time]

    # make an inference
    predicted, expected = predict(cnn, input, target, class_mapping)
    
    
    print(f'Predicted: {predicted}, Expected: {expected}')