import os
import wave
import struct
import uuid
import numpy as np


# MIC settings
nFFT = 1024
CHANNELS = 1
MAX_AMPLITUDE = 32767
# sample rate - count of samples per seconds
RATE = 44100
MAX_FRAGMENT_LENGTH = RATE * 1.9
MIN_FRAGMENT_LENGTH = RATE / 0.6
# on how much (in percent) amplitude shoulb be upper silence to determinate start of fragment
THRESHOLD_OF_SILENCE = 0.8

ASSETSS_FOLDER = '../assetss'


class Fragmenter:
    def __init__(self, source_file):
        self.sample_size = None
        self.fragment = []
        self.rare_fragment = []
        self.noise = []
        self.rare_noise = []
        self.source_file = source_file

    def create_folder(self, directory: str):
        if not os.path.exists(directory):
            os.makedirs(directory)

    def write_chunk(self, chunk, type='fragments'):
        file_path = os.path.join(ASSETSS_FOLDER, 'output', type)
        self.create_folder(file_path)
        file_name = os.path.join(
            ASSETSS_FOLDER, 'output', type, '{}.wav'.format(uuid.uuid4()))
        print('--> write to:', file_name)
        wav_file = wave.open(file_name, 'w')
        wav_file.setparams(
            (self.source_file.getnchannels(), self.source_file.getsampwidth(), self.source_file.getframerate(), self.source_file.getnframes(), "NONE", "not compressed"))
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
            self.write_chunk(self.rare_fragment, 'fragments')
            self.write_chunk(self.rare_noise, 'noise')
            self.clear_fragment()
            self.clear_noise()
        else:
            self.save_noise(amplitude_chunk=chunk, rare_chunk=y)
            self.clear_fragment()


def main():
    file_path = os.path.join(ASSETSS_FOLDER, 'input', 'breath.wav')
    print('--> read:', file_path)
    wav_file = wave.open(file_path, 'rb')
    fragmenter = Fragmenter(wav_file)
    data = wav_file.readframes(nFFT)

    while data != b'':
        fragmenter.find_fragment(data)
        data = wav_file.readframes(nFFT)


main()
