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

# Cuda GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Define a function that converts MIDI file to notes sequences for each MIDI file and maps them to tensor-like objects
def midi_to_notes(midi_file):
    midi_data = pretty_midi.PrettyMIDI(midi_file)
    notes = []
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            notes.append([note.start, note.end, note.pitch, note.velocity])
    return np.array(notes), midi_data

# Load the MAESTRO dataset
data_dir = pathlib.Path('data/maestro-v2.0.0')
if not data_dir.exists():
  tf.keras.utils.get_file(
      'maestro-v2.0.0-midi.zip',
      origin='https://storage.googleapis.com/magentadata/datasets/maestro/v2.0.0/maestro-v2.0.0-midi.zip',
      extract=True,
      cache_dir='.', cache_subdir='data',
  )

#midi_files = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.endswith('.midi')]
midi_files = [str(file) for file in data_dir.glob('**/*.midi')]

midi_files = midi_files[:100] # Use a small subset of the dataset

# Preprocessing
all_notes = []
midi_data_objs = []
for midi_file in midi_files:
  notes, midi_data = midi_to_notes(midi_file)
  if notes.size > 0:
    all_notes.append(notes)
    midi_data_objs.append(midi_data)

if not all_notes:
    raise ValueError("No notes were extracted from the MIDI files. Please check the MIDI files and the midi_to_notes function.")

# Normalize the data
scaler = StandardScaler()
all_notes = np.vstack(all_notes)
scaler.fit(all_notes)
all_notes_scaled = scaler.transform(all_notes)

# Prepare the data for training
sequence_length = 50
X = []
y = []

for i in range(len(all_notes_scaled) - sequence_length):
    X.append(all_notes_scaled[i:i + sequence_length])
    y.append(all_notes_scaled[i + sequence_length])

X = np.array(X)
y = np.array(y)


# Create the NN model
model = models.Sequential([
    layers.LSTM(128, input_shape=(sequence_length, 4), return_sequences=True),
    layers.LSTM(128),
    layers.Dense(128, activation='relu'),
    layers.Dense(4)
])

model.compile(optimizer='adam', loss='mse')


# Train
model.fit(X, y, epochs=50, batch_size=64)
model.save('score.h5')

