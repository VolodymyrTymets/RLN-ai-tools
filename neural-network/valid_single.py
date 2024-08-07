import os
from os import listdir
from os.path import isfile, join
import pathlib
import random

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import tensorflow as tf

RATE = 44100
FRAGMENT_LENGTH = RATE * 2

def get_files(dir_path): 
  return [f for f in listdir(dir_path) if isfile(join(dir_path, f)) and f != '.DS_Store']

def get_wave(file_full_path):
  x = tf.io.read_file(str(file_full_path))
  x, sample_rate = tf.audio.decode_wav(x, desired_channels=1, desired_samples=FRAGMENT_LENGTH,)
  x = tf.squeeze(x, axis=-1)
  return x[tf.newaxis,...]

DATASET_PATH = 'assetss'
valid_dir_path = os.path.join(DATASET_PATH, 'data-set', 'valid')
b_dir_path = os.path.join(valid_dir_path, 'breath')
n_dir_path = os.path.join(valid_dir_path, 'noise')
s_dir_path = os.path.join(valid_dir_path, 'stimulation')
model_dir = pathlib.Path(os.path.join(DATASET_PATH, 'rln-model_2'))


b_files = get_files(b_dir_path)
n_files = get_files(n_dir_path)
s_files = get_files(s_dir_path)
b_files.sort()
n_files.sort()
s_files.sort()
b_names = []
n_names = []
s_names = []
b_waves = []
n_waves = []
s_waves = []

model = tf.saved_model.load(model_dir)

for file in b_files:
  file_full_path = os.path.join(b_dir_path, file)
  waveform = get_wave(file_full_path)
  b_waves.append(waveform);
  b_names.append(file)

for file in n_files:
  file_full_path = os.path.join(n_dir_path, file)
  waveform = get_wave(file_full_path)
  n_waves.append(waveform);
  n_names.append(file)

for file in s_files:
  file_full_path = os.path.join(s_dir_path, file)
  waveform = get_wave(file_full_path)
  s_waves.append(waveform);
  s_names.append(file)  



file_indexs = [0, 1, 4, 7]
for file_index in file_indexs:
  ## Plot
  rows = 3
  cols = 2

  # file_index = random.randint(0, len(s_waves))
  data_set = [n_waves, b_waves, s_waves]
  data_set_file_names = [n_names, b_names, s_names]

  fig, axes = plt.subplots(rows, cols, figsize=(6, 4), gridspec_kw={'width_ratios': [4, 1], 'hspace': 0.5})
  to_perc = np.vectorize(lambda x: x * 100)

  for r in range(rows):
      w_ax = axes[r][0]
      p_ax = axes[r][1]

      wave = data_set[r][file_index]
      file_name = data_set_file_names[r][file_index]
      print('prediction for file:', file_name)
      
      result = model(tf.constant(wave))
      prediction = result['predictions']
      print('prediction:', tf.nn.softmax(prediction[0]))
      # wave
      ax = wave.numpy()[0]
      ax = ax * np.hamming(len(ax))
      w_ax.plot(ax)
      w_ax.set_ylim([-1.1, 1.1])
      w_ax.set_ylabel(file_name, fontweight ='bold')
      w_ax.xaxis.set_major_locator(ticker.NullLocator())
      # w_ax.set_title(str(file_name))

      # Prediction
      label_names = np.array(result['label_names'])
      yticks = to_perc(tf.nn.softmax(prediction[0]))
      # ylabels = [f'{y:1.0f}%' for y in yticks]
      p_ax.bar(['b', 'n', 's'], yticks)

      # p_ax.set_yticks(yticks, labels=ylabels)
      # p_ax.set_ylabel('Prediction (%)', fontweight ='bold')
      # p_ax.set_title(str(file_name))

  plt.show()
