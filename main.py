import numpy as np
import pretty_midi
import os
import matplotlib.pyplot as plt
import librosa.display
import pathlib
from sklearn.preprocessing import StandardScaler
from IPython.display import Audio, display
import tensorflow as tf
import keras
from keras import models, layers


model = tf.keras.models.load_model('score.h5')

def generate_music(model, seed, length):
  generated = []
  current_sequence = seed

  for i in range(length):
    prediction = model.predict(current_sequence[np.newaxis, :, :])[0]
    generated.append(prediction)
    current_sequence = np.vstack([current_sequence[1:], prediction])

  return np.array(generated)

# Convert the generated from tensor -> MIDI
def to_midi(tensor, output_file):
  tensor = scaler.inverse_transform(tensor)
  midi = pretty_midi.PrettyMIDI()
  instrument = pretty_midi.Instrument(program=0)

  for note_data in tensor:
    start, end, pitch, velocity = note_data
    note = pretty_midi.Note(
        velocity = int(velocity),
        pitch = int(pitch),
        start = start,
        end = end
    )
    instrument.notes.append(note)

  midi.instruments.append(instrument)
  midi.write(output_file)

#seed_index = random.randint(0, len(X) -1)
seed = X[25] # The seed for generating music
generated_music = generate_music(model, seed, length=100) # length is the number of notes to generate


# Convert to MIDI
to_midi(generated_music, 'generated_music.mid')

def plot_piano_roll(midi_data, start_time=0, end_time=None):
  for instrument in midi_data.instruments:
    piano_roll = instrument.get_piano_roll()
    plt.figure(figsize=(10,4))
    librosa.display.specshow(piano_roll, sr=midi_data.get_end_time(), hop_length=1, x_axis='time', y_axis='cqt_note')
    plt.title(f'Piano Roll for {instrument.name}')
    plt.show()

  midi_data = pretty_midi.PrettyMIDI('generated_music.mid')
  plot_piano_roll(midi_data)

# Save the music
to_midi(generated_music, 'generated_music_final.mid')
