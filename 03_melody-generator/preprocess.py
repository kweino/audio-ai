import time
import os
import json
from dotenv import load_dotenv
import numpy as np
import music21 as m21
from tensorflow import keras



# Load the environment variables from the project .env file in the parent folder
dotenv_path = os.path.join(os.path.dirname(__file__), '../', '.env')
load_dotenv(dotenv_path)
# Set the environment variable using os.getenv()
INPUT_FILE_PATH = os.getenv('KERN_DATASET_PATH')
INPUT_FILE_TYPE = 'krn' # Type of music file being imported, ex. 'krn' or 'mid'

SAVE_DIR = 'dataset'
SINGLE_FILE_DATASET = 'magyar_dataset_001'
SEQUENCE_LENGTH = 64 # 
MAPPING_PATH = 'mapping.json'

ACCEPTABLE_DURATIONS = [
    0.25,
    0.5,
    0.75,
    1,
    1.5,
    2,
    3,
    4
]


def load_original_songs(dataset_path,input_file_type):
    print(f'Path:{dataset_path}')
    songs = []

    # load all files with music21
    for path, _, files in os.walk(dataset_path):
        for file in files:
            if file[-3:] == input_file_type:
                try:
                    song = m21.converter.parse(os.path.join(path,file))
                    songs.append(song)
                    print(f'{file} loaded')
                except IndexError:
                    print(f'{file} skipped due to Index error.')
                    continue
    return songs

def has_acceptable_durations(song, acceptable_durations):
    for note in song.flat.notesAndRests:
        if note.duration.quarterLength not in acceptable_durations:
            return False
    return True

def transpose(song):
    '''transposes song to Cmaj / Am'''
    # get key from song, if notated 
    parts = song.getElementsByClass(m21.stream.Part)
    measures_part0 = parts[0].getElementsByClass(m21.stream.Measure)
    key = measures_part0[0][4] # location of key in music21 songs

    # if key not notated, estimate using music21
    if not isinstance(key, m21.key.Key):
        key = song.analyze('key')

    # print(f'Key:{key}')

    # get transposition interval
    if key.mode == 'major':
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch('C'))
    elif key.mode == 'minor':
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch('A'))

    # transpose song by transposition interval
    transposed_song = song.transpose(interval)

    print('Songs transposed.')
    return transposed_song

def encode_song(song, time_step=0.25):
    '''encodes song into a time series format'''
    # pitch = 60, duration = 1.0 -> [60, "_", "_", "_"]
    
    encoded_song = []

    for event in song.flat.notesAndRests: # get all notes and rests

        # handle notes
        if isinstance(event, m21.note.Note):
            symbol = event.pitch.midi 
        elif isinstance(event, m21.note.Rest):
            symbol = 'r'

        # convert into time series notation
        steps = int(event.duration.quarterLength / time_step)
        for step in range(steps):
            if step == 0:
                encoded_song.append(symbol)
            else:
                encoded_song.append('_')

    # cast encoded song to a str            
    encoded_song = ' '.join(map(str, encoded_song))

    return encoded_song

def preprocess(dataset_path, input_file_type):
    '''
    Preprocessing Steps
    1. load folk songs
    2. filter out songs that have unacceptable durations
    3. transpose songs to C / Am
    4. encode songs with music time series representation
    5. save songs to text file
    '''
    # load songs
    print('Loading songs...')
    songs = load_original_songs(dataset_path, input_file_type)
    print(f'Loaded {len(songs)} songs.')

    for i, song in enumerate(songs):

        # filter out songs that have unacceptable durations
        if not has_acceptable_durations(song, ACCEPTABLE_DURATIONS):
            print(f'Song {i} had unacceptable duration values and was rejected.')
            continue

        # transpose songs to C / Am
        song = transpose(song)

        # encode songs with music time series representation
        encoded_song = encode_song(song)

        # save songs to text file
        save_path = os.path.join(SAVE_DIR, f'song_{i}')
        with open(save_path, 'w') as fp:
            fp.write(encoded_song)

def load_song(file_path):
    with open(file_path, 'r') as fp:
        song = fp.read()
    return song

def create_single_file_dataset(input_dataset_path, dataset_output_path, sequence_length):
    '''concatenates all the songs into one text file, adding song delimiters'''
    new_song_delimiter = '/ ' * sequence_length # generates slashes for an entire music sequence
    songs = ''

    # load encoded songs and add delimiters
    for path, _, files in os.walk(input_dataset_path):
        for file in files:
            file_path = os.path.join(path, file)
            song = load_song(file_path)
            songs = songs + song + ' ' + new_song_delimiter
            print(f'{file} encoded')

    songs = songs[:-1] # get rid of the final space character in the song list so the length is correct

    # save string containing the full dataset
    with open(dataset_output_path, 'w') as fp:
        fp.write(songs)
    
    return songs

def create_mapping(songs, mapping_path):
    '''creates a json file to map for the symbols present in the song onto integers'''
    mappings = {}

    # id vocabulary
    songs = songs.split() # split on spaces
    vocabulary = list(set(songs)) # use a set to find unique values in the list

    # create mappings
    for i,  symbol in enumerate(vocabulary):
        mappings[symbol] = i

    # save vocab to json file
    with open(mapping_path, 'w') as fp:
        json.dump(mappings, fp)

def convert_songs_to_int(songs):
    #load mappings
    with open(MAPPING_PATH, 'r') as fp:
        mappings = json.load(fp)
        
    # cast songs string to a list
    songs = songs.split()

    # map songs to int
    int_songs = [mappings[symbol] for symbol in songs]

    return int_songs

def generate_training_sequences(sequence_length):
    # [11, 12, 13, 14, ...] -> (input: [11,12], target: 13), ([12,13], 14), ...

    inputs = []
    targets = []

    # load songs & map to int
    songs = load_song(SINGLE_FILE_DATASET)
    int_songs = convert_songs_to_int(songs)
    
    # generate training sequences
    num_sequences = len(int_songs) - sequence_length

    for i in range(num_sequences):
        inputs.append(int_songs[i:i+sequence_length])
        targets.append(int_songs[i+sequence_length])

    # one-hot encode the sequences
    vocab_size = len(set(int_songs))
    inputs = keras.utils.to_categorical(inputs, num_classes=vocab_size)
    targets = np.array(targets)

    print(f'There are {len(inputs)} sequences')

    return inputs, targets




def main():
    print('Preprocessing...')
    try:
        load_song(SINGLE_FILE_DATASET)
        print('Single file found.')
    except OSError:

        preprocess(INPUT_FILE_PATH, INPUT_FILE_TYPE)

        print('Preprocessing complete. Concatenating files...')
        songs = create_single_file_dataset(SAVE_DIR, SINGLE_FILE_DATASET, SEQUENCE_LENGTH)
        print('Files concatenated.')
    
    print('Creating mapping...')
    create_mapping(songs, MAPPING_PATH)
    
    print('Generating training sequences...')
    inputs, targets = generate_training_sequences(SEQUENCE_LENGTH)

if __name__ == '__main__':
    # us = m21.environment.UserSettings()
    # us['musicxmlPath'] = '/Applications/MuseScore 3.app'
    t1 = time.time()
    main()
    t2 = time.time()
    print(f'Preprocessing completed in {round(t2-t1, 5)}s')

    


    