import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


DATASET_PATH = 'assetss'
model_dir = pathlib.Path(os.path.join(DATASET_PATH, 'rln-model'))


# Get validation data 
file_name = 'b4.wav'
x = os.path.join(DATASET_PATH, 'valid', file_name)
x = tf.io.read_file(str(x))
x, sample_rate = tf.audio.decode_wav(x, desired_channels=1, desired_samples=44100,)
x = tf.squeeze(x, axis=-1)
waveform = x[tf.newaxis,...]

# Valide via model
model = tf.saved_model.load(model_dir)
result = model(tf.constant(waveform))
prediction = result['predictions']

# Form result
label_names = np.array(result['label_names'])
plt.bar(label_names, tf.nn.softmax(prediction[0]))
plt.title('Validatoin of {}'.format(file_name))
plt.show()

