import sys
import os
from os import listdir
from os.path import isfile, join
import wave
import librosa
from src.fragmenter.fragmenter import Fragmenter
from src.filter.filter import filter_fragments


ASSETS_FOLDER = 'assets'
frame_length = 2048
hop_length = 64


def fragment_folder(folder_name: str):
  valid_dir_path = os.path.join(ASSETS_FOLDER, 'input', folder_name)
  only_files = [f for f in listdir(valid_dir_path) if isfile(join(valid_dir_path, f)) and f != '.DS_Store']

  for file in only_files:
    file_path = os.path.join(valid_dir_path, file)
    print('--> read:', file_path)
    wav_file = wave.open(file_path, 'rb')
    fragmenter = Fragmenter(wav_file, file)
    data = wav_file.readframes(frame_length * 4)

    while data != b'':
      fragmenter.find_fragment(data)
      data = wav_file.readframes(frame_length * 4)


def main():
  fragment_folder(sys.argv[1])
  filter_fragments(ASSETS_FOLDER, 'noise', frame_length, hop_length)
  filter_fragments(ASSETS_FOLDER, 'fragments', frame_length, hop_length)

main()