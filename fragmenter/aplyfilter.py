import os
from os import listdir
from os.path import isfile, join
import scipy.signal
import scipy.io.wavfile
import wave
import numpy as np
import tensorflow_io as tfio
import pyloudnorm as pyln

import struct

DURATION = 2

ASSETSS_FOLDER = 'assets'

class Filter:
    def __init__(self, out_folder: str, edges: list[float]):
        self.file_name = None
        self.out_folder = out_folder
        self.edges = edges
    
    def get_only_files(self, path):
        return [f for f in listdir(path) if isfile(join(path, f)) and f != '.DS_Store']
    
    def create_folder(self, directory: str):
        if not os.path.exists(directory):
            os.makedirs(directory)

    def thim_noise(self, waveform):
        position = tfio.audio.trim(waveform, axis=0, epsilon=0.1).numpy()
        return waveform[position[0]:position[1]]
        
    def bandpass(self, data: np.ndarray, sample_rate: float, poles: int = 5):
        sos = scipy.signal.butter(poles, self.edges, 'bandpass', fs=sample_rate, output='sos')
        filtered_data = scipy.signal.sosfiltfilt(sos, data)
        return filtered_data
    
    def normalize(self, data: np.ndarray, sample_rate: float):
        meter = pyln.Meter(sample_rate) # create BS.1770 meter
        loudness = meter.integrated_loudness(data) # measure loudness
        # print('---l->', loudness)
        # print('---r->', loudness + 14.0)
        # return pyln.normalize.loudness(data, loudness, loudness + 14.0)
        # peak normalize audio to -1 dB
        return pyln.normalize.peak(data, -1.0)
    
    def buffer_to_float_16(self, audio_as_np_float32):                                                
        max_int16 = 2**15
        audio_normalised = audio_as_np_float32 / max_int16
        return audio_normalised;

    def buffer_to_float_32(self, audio_as_np_float32):                                                 
        max_int16 = 2**15
        audio_normalised = audio_as_np_float32 * max_int16
        return audio_normalised;
    
    def write_to_file(self, chunk, file_name, source_file):
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
        

def apply_for_folder(data_set_path, out_data_set_path, folder_name, edges: list[float]):
    path = os.path.join(data_set_path, folder_name)
    filter = Filter(os.path.join(out_data_set_path, folder_name), edges)
    files = filter.get_only_files(path)
    for file in files:
        file_path = os.path.join(path, file)
        sample_rate, data = scipy.io.wavfile.read(file_path)
        normalized = filter.normalize(data=filter.buffer_to_float_16(data), sample_rate=sample_rate)
    
        filtered = filter.bandpass(filter.buffer_to_float_32(normalized), sample_rate);
        trimmed = filter.thim_noise(filter.buffer_to_float_16(filtered))

        filter.write_to_file(filter.buffer_to_float_32(trimmed), file, wave.open(file_path, 'rb'))


data_set_path_t = os.path.join(ASSETSS_FOLDER, 'data_set_{}s'.format(DURATION), 'train')
out_data_set_path_t = os.path.join(ASSETSS_FOLDER, 'data_set_{}s_n_f'.format(DURATION), 'train')
apply_for_folder(data_set_path_t, out_data_set_path_t, 'breath', [12, 20])
apply_for_folder(data_set_path_t, out_data_set_path_t, 'stimulation', [50, 200])

data_set_path_v = os.path.join(ASSETSS_FOLDER, 'data_set_{}s'.format(DURATION), 'valid')
out_data_set_path_v = os.path.join(ASSETSS_FOLDER, 'data_set_{}s_n_f'.format(DURATION), 'valid')
apply_for_folder(data_set_path_v, out_data_set_path_v, 'breath', [12, 20])
apply_for_folder(data_set_path_v, out_data_set_path_v, 'stimulation', [50, 200])