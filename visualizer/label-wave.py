import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import pathlib
from matplotlib.lines import Line2D
import pyloudnorm as pyln

DATASET_PATH = 'assets'
nFFT = 512
RATE = 44100
FRAGMENT_LENGTH = int(RATE / 5)
DURATION = 200
print('--->', FRAGMENT_LENGTH)

model_dir = pathlib.Path(os.path.join(DATASET_PATH, 'rln-model_{}'.format(DURATION)))
print('--model_dir-->', model_dir)
model = tf.saved_model.load(model_dir)

def to_chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def get_chank_label_by_model(wave):
    x = tf.convert_to_tensor(wave, dtype=tf.float32)
    waveform =  x[tf.newaxis,...]
    result = model(tf.constant(waveform))
    label_names = np.array(result['label_names'])
    prediction = tf.nn.softmax(result['predictions']).numpy()[0]
    max_value = max(prediction)
    i, = np.where(prediction == max_value)
    wave_label = label_names[i]
    return wave_label     
  
def normalize(data: np.ndarray, sample_rate: float):
    meter = pyln.Meter(sample_rate) # create BS.1770 meter
    loudness = meter.integrated_loudness(data) # measure loudness
    return pyln.normalize.peak(data, -1.0)
     

file_path = os.path.join(DATASET_PATH, 'test', 'test5.wav')
x = tf.io.read_file(str(file_path))
x, sample_rate = tf.audio.decode_wav(x, desired_channels=1, desired_samples=RATE * 16)
x = tf.squeeze(x, axis=-1)
x = x[tf.newaxis,...]
_waveform = x.numpy()[0];
waveform = normalize(data=_waveform, sample_rate=RATE) 
waveform = _waveform

chunks = [x for x in to_chunks(_waveform, int(FRAGMENT_LENGTH))]
chunks_n = to_chunks(waveform, int(FRAGMENT_LENGTH))

# Form segments for collection of lines   
segments = []
linecolors = []
x = 0
for i, chunk_n in enumerate(chunks_n):
  lineN = []
  lin_y = chunks[i]
  for i, y in enumerate(lin_y):
    lineN.append((x, y)) 
    x = x + 1
  segments.append(lineN)
  try:
    line_label = get_chank_label_by_model(chunk_n)
  except:
     line_label = 'noise'
  color = 'red' if 'stimulation' in str(line_label) else 'blue'
  color = 'green' if 'breath' in str(line_label) else color
  linecolors.append(color)

print('Labe with duration {}: {}'.format(DURATION, file_path))
# Create figure
fig, ax = plt.subplots(figsize=(12, 2))
line_collection = LineCollection(segments=segments, colors=linecolors)
# Add a collection of lines
ax.add_collection(line_collection)

# Set x and y limits... sadly this is not done automatically for line
# collections
ax.set_xlim(0, len(waveform))
ax.set_ylim(1, -1)
ax.legend([Line2D([0, 1], [0, 1], color='blue'), Line2D([0, 1], [0, 1], color='green'), Line2D([0, 1], [0, 1], color='red')], ['Noise', 'Breath', 'Stimulation'])
plt.show()