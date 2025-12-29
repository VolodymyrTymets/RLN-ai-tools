import librosa
import numpy as np
from scipy import signal as sp_signal


class TimeDomainFeatures:

  def __init__(self):
    pass

  def duration(self, signal: np.ndarray, sr: int):
    return 1 / sr * len(signal)

  def _frame(self, signal: np.ndarray, frame_length: int, hop_length: int):
    return librosa.util.frame(signal, frame_length=frame_length, hop_length=hop_length, axis=0)

  def amplitude_envelope(self, signal: np.ndarray, frame_length: int, hop_length: int):
    return np.array([np.max(np.abs(frame)) for frame in self._frame(signal, frame_length, hop_length)])

  def root_mean_square_energy(self, signal: np.ndarray, frame_length: int, hop_length: int):
    frames = self._frame(signal, frame_length, hop_length)
    return np.sqrt(np.mean(frames ** 2, axis=0))

  def zero_crossing_rate(self, signal: np.ndarray, frame_length: int, hop_length: int):
    return librosa.feature.zero_crossing_rate(signal, frame_length=frame_length, hop_length=hop_length)[0]

  def AE(self, signal: np.ndarray, frame_length: int, hop_length: int):
    return self.amplitude_envelope(signal=signal, frame_length=frame_length, hop_length=hop_length)

  def RSME(self, signal: np.ndarray, frame_length: int, hop_length: int):
    return self.root_mean_square_energy(signal=signal, frame_length=frame_length, hop_length=hop_length)

  def ZCR(self, signal: np.ndarray, frame_length: int, hop_length: int):
    return self.zero_crossing_rate(signal=signal, frame_length=frame_length, hop_length=hop_length)


class FrequencyDomainFeatures:

  def __init__(self):
    pass

  def _freq_for_magnitude(self, magnitude: np.array, sr: int):
    return np.linspace(0, sr, len(magnitude))

  def _magnitude(self, signal: np.array, f_ration: float = 0.5):
    ft = np.fft.fft(signal)
    magnitude = np.abs(ft)
    nup_freq_bins = int(len(magnitude) * f_ration)
    return magnitude[:nup_freq_bins]

  def fft(self, signal: np.ndarray, sr: int, frame_length: int, hop_length: int):
    # Step 1 framing
    frames = librosa.util.frame(signal, frame_length=frame_length, hop_length=hop_length, axis=0)
    # Step 2 windowing
    window = sp_signal.windows.hamming(frame_length)
    # Step 3 FFT
    magnitudes = [self._magnitude(frame * window, 0.5) for frame in frames]
    # Step 4 Aggregation
    magnitude = np.mean(magnitudes, axis=0)
    frequency = self._freq_for_magnitude(magnitude, sr)
    return magnitude, frequency

  def stft(self, signal: np.ndarray, frame_length: int, hop_length: int):
    s_scale = librosa.stft(signal, n_fft=frame_length, hop_length=hop_length)
    return np.abs(s_scale) ** 2
