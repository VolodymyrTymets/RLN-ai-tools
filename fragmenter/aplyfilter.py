import os
from os import listdir
from os.path import isfile, join
import scipy.signal
import scipy.io.wavfile
import wave
import numpy as np
import pathlib
import tensorflow as tf
import shutil
import struct


# MIC settings
nFFT = 512
# sample rate - count of samples per seconds
RATE = 44100
FRAGMENT_LENGTH = int(RATE / 10)
DURATION = 2

ASSETSS_FOLDER = 'assets'

def get_only_files(path):
    return [f for f in listdir(path) if isfile(join(path, f)) and f != '.DS_Store']

class Filter:
    def __init__(self, out_folder):
        self.file_name = None
        self.out_folder = out_folder
    

    def create_folder(self, directory: str):
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    def bandpass(self, data: np.ndarray, edges: list[float], sample_rate: float, poles: int = 5):
        sos = scipy.signal.butter(poles, edges, 'bandpass', fs=sample_rate, output='sos')
        filtered_data = scipy.signal.sosfiltfilt(sos, data)
        return filtered_data
    
    def write_chunk(self, chunk, file_name, source_file):
        self.create_folder(self.out_folder)
        print('--> write to:', file_name)
        wav_file = wave.open(os.path.join(self.out_folder, file_name), 'w')
        wav_file.setparams(
            (1, source_file.getsampwidth(), source_file.getframerate(), source_file.getnframes(), "NONE", "not compressed"))
        for sample in chunk:
            if sample > 0:
                wav_file.writeframes(struct.pack('h', int(sample) if sample <= 32767 else 32767))
            else:
                wav_file.writeframes(struct.pack('h', int(sample) if sample >= -32767 else -32767))


        

def main(path):
    b_path = os.path.join(path, 'breath')
    s_path = os.path.join(path, 'stimulation')

    b_filter = Filter(os.path.join(b_path, 'filtered'))
    s_filter = Filter(os.path.join(s_path, 'filtered'))

    b_files = get_only_files(b_path)
    s_files = get_only_files(s_path)

    for file in s_files:
        file_path = os.path.join(s_path, file)
        wav_file = wave.open(file_path, 'rb')
        sample_rate, data = scipy.io.wavfile.read(file_path)
        filtered = s_filter.bandpass(data, [50, 200], sample_rate);
        s_filter.write_chunk(filtered, file, wav_file)

    # for file in s_files:
    #     file_path = os.path.join(s_path, file)
    #     wav_file = wave.open(file_path, 'rb')
    #     sample_rate, data = scipy.io.wavfile.read(file_path)
    #     filtered = s_filter.bandpass(data, [12, 20], sample_rate);
    #     s_filter.write_chunk(filtered, file, wav_file)



main(os.path.join(ASSETSS_FOLDER, 'data_set_{}s'.format(DURATION), 'train'))