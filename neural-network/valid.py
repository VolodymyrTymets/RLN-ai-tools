import os
from os import listdir
from os.path import isfile, join
import pathlib
import random

import matplotlib.pyplot as plt
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

b_prediction = []
n_prediction = []
s_prediction = []

for wave in b_waves:
    result = model(tf.constant(wave))
    prediction = result['predictions']
    res = tf.nn.softmax(prediction[0]).numpy()
    b_prediction.append(tf.nn.softmax(prediction[0]).numpy()[0])
for wave in n_waves:
    result = model(tf.constant(wave))
    prediction = result['predictions']
    res = tf.nn.softmax(prediction[0]).numpy()
    n_prediction.append(tf.nn.softmax(prediction[0]).numpy()[1])  

for wave in s_waves:
    result = model(tf.constant(wave))
    prediction = result['predictions']
    res = tf.nn.softmax(prediction[0]).numpy()
    s_prediction.append(tf.nn.softmax(prediction[0]).numpy()[2])  

total_n = np.sum(n_prediction) / len(n_prediction) * 100
total_b = np.sum(b_prediction) / len(b_prediction) * 100
total_s = np.sum(s_prediction) / len(s_prediction) * 100


cut_ext = np.vectorize(lambda f: f.replace('.wav', ''))
to_perc = np.vectorize(lambda x: x * 100)

fig, (ax_n, ax_b, ax_s) = plt.subplots(1, 3)
ax_X = range(len(n_prediction))
ax_n.bar(cut_ext(n_files), to_perc(n_prediction))
ax_n.set_ylabel('Prediction (%)', fontweight ='bold')
ax_n.set_xlabel('File (.wav)', fontweight ='bold')
ax_n.set_title('noise {}%'.format(round(total_n, 2)))

ax_b.bar(cut_ext(b_files), to_perc(b_prediction))
ax_b.set_xlabel('File (.wav)', fontweight ='bold')
ax_b.set_title('breat {}%'.format(round(total_b, 2)))

ax_s.bar(cut_ext(s_files), to_perc(s_prediction))
ax_s.set_xlabel('File (.wav)', fontweight ='bold')
ax_s.set_title('stimulation {}%'.format(round(total_s, 2)))


plt.show()

