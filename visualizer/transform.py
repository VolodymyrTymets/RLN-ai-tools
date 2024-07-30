import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import tensorflow as tf
import tensorflow_io as tfio

DATASET_PATH = 'assets'
nFFT = 512
RATE = 44100

def get_spectrogram(waveform):
  # Convert the waveform to a spectrogram via a STFT.
  spectrogram = tf.signal.stft(
      waveform, frame_length=256, frame_step=128)

  # Obtain the magnitude of the STFT.
  spectrogram = tf.abs(spectrogram)
  # Add a `channels` dimension, so that the spectrogram can be used
  # as image-like input data with convolution layers (which expect
  # shape (`batch_size`, `height`, `width`, `channels`).
  spectrogram = spectrogram[..., tf.newaxis]
  return spectrogram

def plot_spectrogram(spectrogram, ax):
  if len(spectrogram.shape) > 2:
    assert len(spectrogram.shape) == 3
    spectrogram = np.squeeze(spectrogram, axis=-1)
  # Convert the frequencies to log scale and transpose, so that the time is
  # represented on the x-axis (columns).
  # Add an epsilon to avoid taking a log of zero.
  log_spec = np.log(spectrogram.T + np.finfo(float).eps)
  height = log_spec.shape[0]
  width = log_spec.shape[1]
  X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
  Y = range(height)
  ax.pcolormesh(X, Y, log_spec)

def get_waveform(file_name):
  x = os.path.join(DATASET_PATH, file_name)
  x = tf.io.read_file(str(x))
  x, sample_rate = tf.audio.decode_wav(x, desired_channels=1, desired_samples=RATE * 2.5,)
  x = tf.squeeze(x, axis=-1)
  x = x[tf.newaxis,...]
  return x.numpy()[0];

noise_waveform = get_waveform('n.wav')
noise_hamming_waveform = noise_waveform * np.hamming(len(noise_waveform))
noise_spectrogram = get_spectrogram(noise_hamming_waveform).numpy()
noise_data = [noise_waveform, noise_hamming_waveform, noise_spectrogram]


breath_waveform = get_waveform('b.wav')
# Trim noise
b_position = tfio.audio.trim(breath_waveform, axis=0, epsilon=0.1).numpy()
b_trim_wave = breath_waveform[b_position[0]:b_position[1]]
breath_hamming_waveform = b_trim_wave * np.hamming(len(b_trim_wave))
breath_spectrogram = get_spectrogram(breath_hamming_waveform).numpy()
breath_data = [breath_waveform, breath_hamming_waveform, breath_spectrogram]

stimulation_waveform = get_waveform('s.wav')
# Trim noise
s_position = tfio.audio.trim(stimulation_waveform, axis=0, epsilon=0.1).numpy()
s_trim_wave = stimulation_waveform[s_position[0]:s_position[1]]
stimulation_hamming_waveform = s_trim_wave * np.hamming(len(s_trim_wave))
stimulation_spectrogram = get_spectrogram(stimulation_hamming_waveform).numpy()
stimulation_data = [stimulation_waveform, stimulation_hamming_waveform, stimulation_spectrogram]


data_set = [noise_data, breath_data, stimulation_data]
## Plot
rows = len(data_set)
cols = len(data_set[0])
# fig, (w_ax, hm_ax, sg_ax) = plt.subplots(rows, cols)

fig, axes = plt.subplots(rows, cols, figsize=(16, 9))


for r in range(rows):
    w_ax = axes[r][0]
    hm_ax = axes[r][1]
    sg_ax = axes[r][2]
    wave,hamming,spectogram = data_set[r];
    # wave
    w_ax.plot(wave)
    w_ax.set_ylim([-1.1, 1.1])
    w_ax.set_title('Time')

    # Hamming
    hm_ax.plot(hamming)
    hm_ax.set_ylim([-1.1, 1.1])
    hm_ax.set_title('Windowed Time')

    # Spectogram
    plot_spectrogram(spectogram, sg_ax)
    sg_ax.set_title('Time and Frequency')

plt.show()