# SCORE: Sequential Composition and Orchestration with Recurrent Engines

SCORE is my attempt at making an ai generating music tool that uses LSTM nerual networks to compose MIDI tracks using the MAESTRO dataset.
For this example model I only used 100 MIDI files to train the model due to memory limitations so the results are a bit underwhelming

## Basic Rundown:
1. MIDI files are converted to note sequences using the `pretty_midi` library.
  - Each note is stored with its start time, end time, pitch, and velocity
2. The extracted notes are normalized
3. The normalized notes are split into fixed lengths (for my model I used a sequence length of 50)
4. Each sequence is used to train the model where the model tries to predict the next note in the sequence
5. After training, the model uses an initial seed based off the training data to predict the next note in the sequence
6. This process is repeated to generate a sequence of a specified length





