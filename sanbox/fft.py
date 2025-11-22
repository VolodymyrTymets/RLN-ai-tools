import os
import matplotlib.pyplot as plt
import librosa, librosa.display
from scipy import signal
import numpy as np

DATASET_PATH = 'assets'
frame_length = 2048
hop_length = 64

b_1, sr = librosa.load(os.path.join(DATASET_PATH, 'breath', '1.wav'))
b_2, _ = librosa.load(os.path.join(DATASET_PATH, 'breath', '2.wav'))
b_3, _ = librosa.load(os.path.join(DATASET_PATH, 'breath', '3.wav'))

n_1, _ = librosa.load(os.path.join(DATASET_PATH, 'nerve', '1.wav'))
n_2, _ = librosa.load(os.path.join(DATASET_PATH, 'nerve', '2.wav'))
n_3, _ = librosa.load(os.path.join(DATASET_PATH, 'nerve', '3.wav'))


def freq_for_magnitude(magnitude: np.array, sr: int):
  return np.linspace(0, sr, len(magnitude))

def signal_to_spectrogram(signal: np.array, f_ration: float = 1):
  ft = np.fft.fft(signal)
  magnitude = np.abs(ft)
  nup_freq_bins = int(len(magnitude) * f_ration)
  return magnitude[:nup_freq_bins]

def freq_domain_pipline(wave: np.array, sr: int):
  # Step 1 framing
  frames = librosa.util.frame(wave, frame_length=frame_length, hop_length=hop_length, axis=0)
  # Step 2 windowing
  window = signal.windows.hamming(frame_length)
  # Step 3 FFT
  magnitudes = [signal_to_spectrogram(frame * window, 0.5) for frame in frames]
  # Step 4 Aggregation
  magnitude = np.mean(magnitudes, axis=0)
  frequency = freq_for_magnitude(magnitude, sr)
  return magnitude, frequency

def show_spectrogram(magnitude, frequency):
  plt.plot( frequency, magnitude)
  plt.xlabel('Magnitude Hz')
  plt.show()

def get_max_freq(magnitude, frequency):
  return frequency[np.argmax(magnitude)]

breaths = [b_1, b_2, b_3]
nerves = [n_1, n_2, n_3]
for b in breaths:
  magnitude, frequency = freq_domain_pipline(b, sr)
  print('Hz of a signal breath : ', get_max_freq(magnitude, frequency))

for n in nerves:
  magnitude, frequency = freq_domain_pipline(n, sr)
  print('Hz of a signal nerve: ', get_max_freq(magnitude, frequency))
  show_spectrogram(magnitude, frequency)


