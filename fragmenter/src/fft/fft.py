import librosa
from scipy import signal
import numpy as np

def freq_for_magnitude(magnitude: np.array, sr: int):
  return np.linspace(0, sr, len(magnitude))

def signal_to_spectrogram(signal: np.array, f_ration: float = 1):
  ft = np.fft.fft(signal)
  magnitude = np.abs(ft)
  nup_freq_bins = int(len(magnitude) * f_ration)
  return magnitude[:nup_freq_bins]

def freq_domain_pipline(wave: np.array, sr: int, frame_length: int, hop_length: int):
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

def get_max_freq(magnitude, frequency):
  return frequency[np.argmax(magnitude)]