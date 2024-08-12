import os
from os import listdir
from os.path import isfile, join
import wave
import struct
import uuid
import numpy as np
import tensorflow_io as tfio


# MIC settings
nFFT = 1024
CHANNELS = 1
MAX_AMPLITUDE = 32767
# sample rate - count of samples per seconds
RATE = 44100
FRAGMENT_LENGTH = RATE * 2
MAX_FRAGMENT_LENGTH = RATE * 2
MIN_FRAGMENT_LENGTH = RATE / 0.3
# on how much (in percent) amplitude shoulb be upper silence to determinate start of fragment
THRESHOLD_OF_SILENCE = 0.8

ASSETSS_FOLDER = 'assets'



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
        trimed = self.thim_noise(wave)
        try:
            chunks = self.to_chunks(trimed, FRAGMENT_LENGTH)
            for chunk in chunks:
                if(len(chunk) < FRAGMENT_LENGTH):
                    return
                # windowed_chunk= chunk * np.hamming(len(chunk))
                self.write_chunk(chunk, 'fragment')
        except Exception as e:
            print("write_fragment: An exception occurred")
            print(e)
            return
            

    def write_noise(self, wave):
        try:
            chunks = self.to_chunks(wave, FRAGMENT_LENGTH)
            for chunk in chunks:
                if(len(chunk) < FRAGMENT_LENGTH):
                    return
                # windowed_chunk= chunk * np.hamming(len(chunk))
                self.write_chunk(chunk, 'noise')
        except Exception as e:
            print("write_fragment: An exception occurred")
            print(e)
        return     

    def write_chunk(self, chunk, type='fragments'):
        file_path = os.path.join(ASSETSS_FOLDER, 'output', type)
        self.create_folder(file_path)
        file_name = os.path.join(
            ASSETSS_FOLDER, 'output', type, '{}.wav'.format(uuid.uuid4()))
        print('--> write to:', file_name)
        wav_file = wave.open(file_name, 'w')
        wav_file.setparams(
            (1, self.source_file.getsampwidth(), self.source_file.getframerate(), self.source_file.getnframes(), "NONE", "not compressed"))
        for sample in chunk:
            wav_file.writeframes(struct.pack('h', int(sample)))

    def save_fragment(self, amplitude_chunk, rare_chunk):
        self.fragment = np.concatenate((self.fragment, amplitude_chunk))
        self.rare_fragment = np.concatenate((self.rare_fragment, rare_chunk))

    def save_noise(self, amplitude_chunk, rare_chunk):
        self.noise = np.concatenate((self.noise, amplitude_chunk))
        self.rare_noise = np.concatenate((self.rare_noise, rare_chunk))

    def thim_noise(self, waveform):
        b_position = tfio.audio.trim(waveform, axis=0, epsilon=0.1).numpy()
        return waveform[b_position[0]:b_position[1]]

    def clear_fragment(self):
        self.fragment = []
        self.rare_fragment = []

    def clear_noise(self):
        self.noise = []
        self.rare_noise = []

    def find_fragment(self, in_data):
        y = []
        try:
            y = np.array(struct.unpack("%dh" %
                                       (self.source_file.getnchannels() * nFFT), in_data))
        except:
            print("An exception occurred")
            return
        y_L = y[::2]
        y_R = y[1::2]
        chunk = np.hstack((y_L, y_R))
        mean_fragment = np.mean(np.abs(chunk))
        percentage = 0 if mean_fragment == 0 else mean_fragment / \
            (MAX_AMPLITUDE) * 100

        len_fragment = len(self.fragment)
        min_fragment_length = MIN_FRAGMENT_LENGTH
        max_fragment_length = MAX_FRAGMENT_LENGTH

        # find in which position of fragment current chunk
        is_start = (len_fragment < min_fragment_length)
        is_tail = (len_fragment >
                   min_fragment_length and len_fragment < max_fragment_length)
        is_end = (len_fragment >= max_fragment_length)
        if (percentage > THRESHOLD_OF_SILENCE and is_start):
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


def main():
    valid_dir_path = os.path.join(ASSETSS_FOLDER, 'input', '00')
    onlyfiles = [f for f in listdir(valid_dir_path) if isfile(join(valid_dir_path, f)) and f != '.DS_Store']

    for file in onlyfiles:
        file_path = os.path.join(valid_dir_path, file)
        print('--> read:', file_path)
        wav_file = wave.open(file_path, 'rb')   
        fragmenter = Fragmenter(wav_file, file)
        data = wav_file.readframes(nFFT)

        while data != b'':
            fragmenter.find_fragment(data)
            data = wav_file.readframes(nFFT)


main()
