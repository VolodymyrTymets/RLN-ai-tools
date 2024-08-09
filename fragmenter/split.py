import os
from os import listdir
from os.path import isfile, join
import wave
import struct
import uuid
import numpy as np


# MIC settings
nFFT = 512
# sample rate - count of samples per seconds
RATE = 44100
FRAGMENT_LENGTH = int(RATE / 10)
DURATION = round(1 / (RATE / FRAGMENT_LENGTH), 2)


def get_only_files(path):
    return [f for f in listdir(path) if isfile(join(path, f)) and f != '.DS_Store']

class Fragmenter:
    def __init__(self, model, out_folder):
        self.sample_size = None
        self.fragment = []
        self.model = model;
        self.file_name = None
        self.out_folder = out_folder
        self.counter = 0
    

    def create_folder(self, directory: str):
        if not os.path.exists(directory):
            os.makedirs(directory)

    def save_fragment(self, chunk):
        self.fragment = np.concatenate((self.fragment, chunk))
   
    def clear_fragment(self):
        self.fragment = []
    
    def buffer_to_chunk(self, in_data, chanels_count): 
        y = []
        try:
            y = np.array(struct.unpack("%dh" % (chanels_count * nFFT), in_data))
        except:
            print("An exception occurred")
            return
        y_L = y[::2]
        y_R = y[1::2]
        chunk = np.hstack((y_L, y_R))
        return chunk
        
            
    def split(self, buffer, source_file, file_name):
        self.file_name = file_name
        chunk = self.buffer_to_chunk(in_data=buffer, chanels_count=source_file.getnchannels())
        if(chunk is None):
            return
        if(len(self.fragment) <  FRAGMENT_LENGTH):
            self.save_fragment(chunk);
        else:
            self.write_chunk(self.fragment, source_file)
            self.clear_fragment()
            self.save_fragment(chunk)


    def write_chunk(self, chunk, source_file):
        self.create_folder(self.out_folder)
        self.counter = self.counter + 1;
        file_name = os.path.join(self.out_folder, '{}_{}.wav'.format(self.counter, uuid.uuid4()))
        print('--> write to:', file_name)
        wav_file = wave.open(file_name, 'w')
        wav_file.setparams(
            (1, source_file.getsampwidth(), source_file.getframerate(), source_file.getnframes(), "NONE", "not compressed"))
        for sample in chunk:
            wav_file.writeframes(struct.pack('h', int(sample)))


def append_duration(name):
    return '{}_{}'.format(DURATION, name)

def split(path):
    n_path = os.path.join(path, 'noise')
    n_out_path = os.path.join(path, append_duration('noise'))
    b_path = os.path.join(path, 'breath')
    b_out_path = os.path.join(path, append_duration('breath'))
    s_path = os.path.join(path, 'stimulation')
    s_out_path = os.path.join(path, append_duration('stimulation'))
    
    mode = None;
    n_fragmenter = Fragmenter(mode, n_out_path)
    b_fragmenter = Fragmenter(mode, b_out_path)
    s_fragmenter = Fragmenter(mode, s_out_path)

    n_files = get_only_files(n_path)
    b_files = get_only_files(b_path)
    s_files = get_only_files(s_path)

    for file in n_files:
        file_path = os.path.join(n_path, file)
        print('--> read from:', file_path)
        wav_file = wave.open(file_path, 'rb')
        data = wav_file.readframes(nFFT)
        while data != b'':
            n_fragmenter.split(data, wav_file, file)
            data = wav_file.readframes(nFFT)

    for file in b_files:
        file_path = os.path.join(b_path, file)
        print('--> read from:', file_path)
        wav_file = wave.open(file_path, 'rb')
        data = wav_file.readframes(nFFT)
        while data != b'':
            b_fragmenter.split(data, wav_file, file)
            data = wav_file.readframes(nFFT)   

    for file in s_files:
        file_path = os.path.join(s_path, file)
        print('--> read from:', file_path)
        wav_file = wave.open(file_path, 'rb')
        data = wav_file.readframes(nFFT)
        while data != b'':
            s_fragmenter.split(data, wav_file, file)
            data = wav_file.readframes(nFFT)             

ASSETSS_FOLDER = 'assets/data-set'


split(os.path.join(ASSETSS_FOLDER, 'valid'))
split(os.path.join(ASSETSS_FOLDER, 'train'))