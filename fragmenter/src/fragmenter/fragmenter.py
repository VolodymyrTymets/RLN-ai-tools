import os
import numpy as np
import wave
import uuid
import struct
from src.fft.fft import freq_domain_pipline, get_max_freq

# sample rate - count of samples per seconds
RATE = 44100
FRAGMENT_LENGTH = int(RATE * 2.2)
MAX_FRAGMENT_LENGTH = int(RATE * 2.2)
MIN_FRAGMENT_LENGTH = RATE / 0.3
# on how much (in percent) amplitude shoulb be upper silence to determinate start of fragment
THRESHOLD_OF_SILENCE_BY_FREQ = 15
max_int16 = 2 ** 15

ASSETS_FOLDER = 'assets'

frame_length = 2048 * 4
hop_length = 64

class Fragmenter:
  def __init__(self, source_file, file_name):
    self.sample_size = None
    self.fragment = []
    self.rare_fragment = []
    self.noise = []
    self.rare_noise = []
    self.source_file = source_file
    self.file_name = file_name;

  def create_folder(self, directory: str):
    if not os.path.exists(directory):
      os.makedirs(directory)

  def to_chunks(self, lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
      yield lst[i:i + n]

  def write_fragment(self, wave):
    try:
      chunks = self.to_chunks(wave, FRAGMENT_LENGTH)
      for chunk in chunks:
        if (len(chunk) < FRAGMENT_LENGTH):
          return
        self.write_chunk(chunk, 'fragments')
    except Exception as e:
      print("write_fragment: An exception occurred")
      print(e)
      return

  def write_noise(self, wave):
    try:
      chunks = self.to_chunks(wave, FRAGMENT_LENGTH)
      for chunk in chunks:
        if (len(chunk) < FRAGMENT_LENGTH):
          return
        self.write_chunk(chunk, 'noise')
    except Exception as e:
      print("write_fragment: An exception occurred")
      print(e)
    return

  def write_chunk(self, chunk, type='fragments'):
    file_path = os.path.join(ASSETS_FOLDER, 'output', type)
    self.create_folder(file_path)
    file_name = os.path.join(
      ASSETS_FOLDER, 'output', type, '{}.wav'.format(uuid.uuid4()))
    print('--> write to:', file_name)
    wav_file = wave.open(file_name, 'w')
    wav_file.setparams(
      (1, self.source_file.getsampwidth(), self.source_file.getframerate(), self.source_file.getnframes(), "NONE",
       "not compressed"))
    for sample in chunk:
      wav_file.writeframes(struct.pack('h', int(sample)))

  def save_fragment(self, amplitude_chunk, rare_chunk):
    self.fragment = np.concatenate((self.fragment, amplitude_chunk))
    self.rare_fragment = np.concatenate((self.rare_fragment, rare_chunk))

  def save_noise(self, amplitude_chunk, rare_chunk):
    self.noise = np.concatenate((self.noise, amplitude_chunk))
    self.rare_noise = np.concatenate((self.rare_noise, rare_chunk))

  def clear_fragment(self):
    self.fragment = []
    self.rare_fragment = []

  def clear_noise(self):
    self.noise = []
    self.rare_noise = []

  def find_fragment(self, in_data):
    try:
      y = np.array(struct.unpack("%dh" %
                                 (self.source_file.getnchannels() * frame_length), in_data))
    except Exception as e:
      print(e)
      return
    y_L = y[::2]
    y_R = y[1::2]
    chunk = np.hstack((y_L, y_R))
    wave = chunk / max_int16
    mag, freq = freq_domain_pipline(wave, RATE, int(frame_length / 4), hop_length)
    max_freq = get_max_freq(mag, freq)

    len_fragment = len(self.fragment)
    min_fragment_length = MIN_FRAGMENT_LENGTH
    max_fragment_length = MAX_FRAGMENT_LENGTH

    # find in which position of fragment current chunk
    is_start = (len_fragment < min_fragment_length)
    is_tail = (min_fragment_length < len_fragment < max_fragment_length)
    is_end = (len_fragment >= max_fragment_length)
    if (max_freq >  THRESHOLD_OF_SILENCE_BY_FREQ and is_start):
      self.save_fragment(amplitude_chunk=chunk, rare_chunk=y)
    elif (is_tail):
      self.save_fragment(amplitude_chunk=chunk, rare_chunk=y)
    elif (is_end):
      self.write_fragment(self.rare_fragment)
      self.write_noise(self.rare_noise)
      self.clear_fragment()
      self.clear_noise()
    else:
      self.save_noise(amplitude_chunk=chunk, rare_chunk=y)
      self.clear_fragment()