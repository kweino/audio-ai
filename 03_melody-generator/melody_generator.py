import json
import numpy as np
from tensorflow import keras
import music21 as m21

from preprocess import SEQUENCE_LENGTH, MAPPING_PATH


TEMPERATURE = 0.3

class MelodyGenerator:

    def __init__(self, model_path='model.h5'):
        self.model_path = model_path
        self.model = keras.models.load_model(model_path)

        with open(MAPPING_PATH, 'r') as fp:
            self._mappings = json.load(fp)
        
        self._start_symbols = ['/'] * SEQUENCE_LENGTH
    
    def _sample_with_temperature(self, probabilities, temperature):
        predictions = np.log(probabilities) / temperature
        probabilities = np.exp(predictions) / np.sum(np.exp(predictions)) # applies softmax function

        choices = range(len(probabilities))
        index = np.random.choice(choices, p=probabilities)

        return index
    
    def generate_melody(self, seed, num_steps, max_sequence_length, temperature):
        
        # create seed with start symbols
        seed = seed.split()
        melody = seed
        seed = self._start_symbols + seed

        # map seed to integers
        seed = [self._mappings[symbol] for symbol in seed]

        for _ in range(num_steps):
            # limit seed to max_sequence_length
            seed = seed[-max_sequence_length:]

            # OHE seed
            ohe_seed = keras.utils.to_categorical(seed, num_classes=len(self._mappings))
            # (max_sequence_length, num vocab symbols)
            ohe_seed = ohe_seed[np.newaxis, ...]

            # make a prediction
            probs = self.model.predict(ohe_seed)[0]
            
            output_int = self._sample_with_temperature(probs, temperature)

            # update seed
            seed.append(output_int)

            # map int to encoding
            output_symbol = [k for k,v in self._mappings.items() if v == output_int][0]

            # check if we're at the end of the melody
            if output_symbol == '/':
                break

            # update melody
            melody.append(output_symbol)

        return melody

    def save_melody(self, melody, step_duration=0.25, format='midi', file_name='mel.mid'):

        # create a music21 stream
        stream = m21.stream.Stream()

        # parse all symbols in the melody and create note/rest objects
        start_symbol = None
        step_counter = 1

        for i, symbol in enumerate(melody):

            # handle case in which we have a note/rest
            if symbol != '_' or i+1 == len(melody):
                # ensure we're not dealing with the first note event
                if start_symbol is not None:

                    quarter_length_duration = step_duration * step_counter

                    # rests
                    if start_symbol == 'r':
                        m21_event = m21.note.Rest(quarterLength=quarter_length_duration)

                    # notes
                    else:
                        m21_event = m21.note.Note(int(start_symbol), quarterLength=quarter_length_duration)
                    
                    stream.append(m21_event)

                    step_counter = 1

                start_symbol = symbol

            # handle prolongations
            else:
                step_counter += 1


        stream.show()
        # write m21 stream to midi file
        stream.write(format, file_name)
        
if __name__ == '__main__':
    mg = MelodyGenerator()
    seed = '55 _ 60 _ 59 _ 60 _ _ _ _ _'
    melody = mg.generate_melody(seed, 500, SEQUENCE_LENGTH, TEMPERATURE)
    print(melody)

    mg.save_melody(melody)