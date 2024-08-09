import os
from os import listdir
from os.path import isfile, join
import wave
import numpy as np
import pathlib
import tensorflow as tf
import shutil


# MIC settings
nFFT = 512
# sample rate - count of samples per seconds
RATE = 44100
FRAGMENT_LENGTH = int(RATE / 10)
DURATION = round(1 / (RATE / FRAGMENT_LENGTH), 2)

ASSETSS_FOLDER = 'assets'

model_dir = pathlib.Path(os.path.join(ASSETSS_FOLDER, 'rln-model_{}s'.format(DURATION)))
model = tf.saved_model.load(model_dir)


def get_only_files(path):
    return [f for f in listdir(path) if isfile(join(path, f)) and f != '.DS_Store']

class Filter:
    def __init__(self, model, out_folder):
        self.model = model;
        self.file_name = None
        self.out_folder = out_folder
    

    def create_folder(self, directory: str):
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    # Tensorflow used float_32 to train model istead of int 16 so need convert buffer to it
    # https://www.tensorflow.org/api_docs/python/tf/audio/decode_wav
    def buffer_to_float_32(self, buffer):
        # Convert buffer to float32 using NumPy                                                                                 
        audio_as_np_int16 = np.frombuffer(buffer, dtype=np.int16)
        audio_as_np_float32 = audio_as_np_int16.astype(np.float32)

        # Normalise float32 array so that values are between -1.0 and +1.0                                                      
        max_int16 = 2**15
        audio_normalised = audio_as_np_float32 / max_int16
        return audio_normalised;
    
    def get_chank_label_by_model(self, wave):
        x = tf.convert_to_tensor(wave, dtype=tf.float32)
        waveform = x[tf.newaxis,...]
        result = self.model(tf.constant(waveform))
        label_names = np.array(result['label_names'])
        prediction = tf.nn.softmax(result['predictions']).numpy()[0]
        max_value = max(prediction)
        i, = np.where(prediction == max_value)
        wave_label = label_names[i]
        return wave_label

    def filter(self, wave, label, file):
        wave_label = self.get_chank_label_by_model(wave=wave)
        if(label not in str(wave_label)):
            from_path = os.path.join(self.out_folder, file)
            to_path = os.path.join(self.out_folder, str(wave_label), file)
            self.create_folder(os.path.join(self.out_folder, str(wave_label)))
            print('{} -> {}'.format(from_path, to_path))
            shutil.move(from_path, to_path)

        

def filter(path):
    n_path = os.path.join(path, 'noise')
    b_path = os.path.join(path, 'breath')
    s_path = os.path.join(path, 'stimulation')

    
    n_filter = Filter(model, n_path)
    b_filter = Filter(model, b_path)
    s_filter = Filter(model, s_path)

    n_files = get_only_files(n_path)
    b_files = get_only_files(b_path)
    s_files = get_only_files(s_path)

    for file in s_files:
        file_path = os.path.join(s_path, file)
        wav_file = wave.open(file_path, 'rb')
        data = wav_file.readframes(nFFT)
        chunks = []
        while data != b'':
            chunk = s_filter.buffer_to_float_32(data)
            chunks = np.concatenate((chunks, chunk))
            data = wav_file.readframes(nFFT)

        s_filter.filter(wave=chunks[:FRAGMENT_LENGTH], label="stimulation", file=file)

    for file in b_files:
        file_path = os.path.join(b_path, file)
        wav_file = wave.open(file_path, 'rb')
        data = wav_file.readframes(nFFT)
        chunks = []
        while data != b'':
            chunk = b_filter.buffer_to_float_32(data)
            chunks = np.concatenate((chunks, chunk))
            data = wav_file.readframes(nFFT)

        b_filter.filter(wave=chunks[:FRAGMENT_LENGTH], label="breath", file=file)   

    for file in n_files:
        file_path = os.path.join(n_path, file)
        wav_file = wave.open(file_path, 'rb')
        data = wav_file.readframes(nFFT)
        chunks = []
        while data != b'':
            chunk = n_filter.buffer_to_float_32(data)
            chunks = np.concatenate((chunks, chunk))
            data = wav_file.readframes(nFFT)

        n_filter.filter(wave=chunks[:FRAGMENT_LENGTH], label="noise", file=file)         




filter(os.path.join(ASSETSS_FOLDER, 'data_set_{}s'.format(DURATION), 'train'))
# filter(os.path.join(ASSETSS_FOLDER, 'data_set_{}s'.format(DURATION), 'valid'))