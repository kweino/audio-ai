import os
from dotenv import load_dotenv

import torch
from torch import nn
from torch.utils.data import DataLoader
import torchaudio

from urbansounddataset import UrbanSoundDataset
from cnn import CNNNEtwork

BATCH_SIZE = 128
EPOCHS=10
LEARNING_RATE = 0.001

SAMPLE_RATE = 22050
NUM_SAMPLES = 22050

# Load the environment variables from the .env file in the parent folder
dotenv_path = os.path.join(os.path.dirname(__file__), '../', '.env')
load_dotenv(dotenv_path)
# Set the environment variable using os.getenv()
ANNOTATIONS_FILE = os.getenv('ANNOTATIONS_FILE')
AUDIO_DIR = os.getenv('AUDIO_DIR')




def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    return train_dataloader

def train_single_epoch(model, data_loader, loss_fn, optimizer, device):
    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        #calculate loss
        predictions = model(inputs)
        loss = loss_fn(predictions, targets)

        #backpropagate loss and update weights
        optimizer.zero_grad() #reset gradients
        loss.backward() #apply backpropogation
        optimizer.step() #update

    print(f'Loss: {loss.item()}')


def train(model, data_loader, loss_fn, optimizer, device, epochs):
    for i in range(epochs):
        print(f'Epoch {i+1}')
        train_single_epoch(model, data_loader, loss_fn, optimizer, device)
        print('--------------------')
    print('Training done!')

if __name__ == "__main__":

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print(f'Using {device}')
    
    # instantiate mel spectrogram transformation
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    # instantiate dataset
    usd = UrbanSoundDataset(ANNOTATIONS_FILE, 
                            AUDIO_DIR, 
                            mel_spectrogram, 
                            SAMPLE_RATE,
                            NUM_SAMPLES,
                            device)

    #create a data loader for the train set
    train_data_loader = create_data_loader(usd, BATCH_SIZE)

    #build model
  
    cnn = CNNNEtwork().to(device)
    print(cnn)
    
    #instantiate loss function & optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn.parameters(),
                                lr=LEARNING_RATE)

    #train model
    train(cnn, train_data_loader, loss_fn, optimizer, device, epochs=EPOCHS)

    #store model
    torch.save(cnn.state_dict(), 'cnnnet.pth')
    print('Model trained and stored at cnnnet.pth')