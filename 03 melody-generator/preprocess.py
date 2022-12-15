import os
import music21 as m21

'''
Steps
1. load folk songs
2. filter out songs that have unacceptable durations
3. transpose songs to C / Am
4. encode songs with music time series representation
5. save songs to text file
'''

KERN_DATASET_PATH = '/Users/kevinweingarten/Library/CloudStorage/OneDrive-UniversityofKansas/Data Science/Projects/audio-ai/data/magyar'
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

def load_songs_in_kern(dataset_path):

    songs = []

    # load all files with music21
    for path, subdir, files in os.walk(dataset_path):
        for file in files:
            if file[-3:] == 'krn':
                song = m21.converter.parse(os.path.join(path,file))
                songs.append(song)
    return songs

def has_acceptable_durations(song, acceptable_durations):
    for note in song.flat.notesAndRests:
        if note.duration.quarterLength not in acceptable_durations:
            return False
    return True

def transpose(song):

    # get key from song, if notated 
    parts = song.getElementsByClass(m21.stream.Part)
    measures_part0 = parts[0].getElementsByClass(m21.stream.Measure)
    key = measures_part0[0][4] # location of key in music21 songs

    # if key not notated, estimate using music21
    if not isinstance(key, m21.key.Key):
        key = song.analyze('key')

    print(f'Key:{key}')

    # get transposition interval
    if key.mode == 'major':
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch('C'))
    elif key.mode == 'minor':
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch('A'))

    # transpose song by transposition interval
    transposed_song = song.transpose(interval)

    return transposed_song

def preprocess(dataset_path):
    pass

    # load songs
    print('Loading songs...')
    songs = load_songs_in_kern(dataset_path)
    print(f'Loaded {len(songs)} songs.')

    for song in songs:

        # filter out songs that have unacceptable durations
        if not has_acceptable_durations(song, ACCEPTABLE_DURATIONS):
            continue

        # transpose songs to C / Am
        song = transpose(song)
        # encode songs with music time series representation
        # save songs to text file



if __name__ == '__main__':
    us = m21.environment.UserSettings()
    us['musicxmlPath'] = '/Applications/MuseScore 3.app'

    #load songs
    songs = load_songs_in_kern(KERN_DATASET_PATH)
    print(f'Loaded {len(songs)} songs.')
    song = songs[15]

    print(f'Acceptable durations in song? {has_acceptable_durations(song,ACCEPTABLE_DURATIONS)}')
    
    transposed_song = transpose(song)

    song.show()
    transposed_song.show()

    