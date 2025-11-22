import os
from os import listdir
from os.path import isfile, join
import shutil
import librosa

from src.fft.fft import freq_domain_pipline, get_max_freq

def filter_fragments(assets_folder: str, folder_name: str, frame_length: int, hop_length: int ):
  valid_dir_path = os.path.join(assets_folder, 'output', folder_name)

  noise_dir_path = os.path.join(assets_folder, 'output', 'noise')
  breath_dir_path = os.path.join(assets_folder, 'output', 'breath')
  stimulation_dir_path = os.path.join(assets_folder, 'output', 'stimulation')

  if not os.path.exists(breath_dir_path):
    os.makedirs(breath_dir_path)
  if not os.path.exists(stimulation_dir_path):
    os.makedirs(stimulation_dir_path)
    if not os.path.exists(noise_dir_path):
      os.makedirs(noise_dir_path)

  only_files = [f for f in listdir(valid_dir_path) if isfile(join(valid_dir_path, f)) and f != '.DS_Store']

  for file in only_files:
    file_path = os.path.join(valid_dir_path, file)
    wave, sr = librosa.load(os.path.join(file_path))
    freq, mag = freq_domain_pipline(wave, sr, frame_length, hop_length)
    max = get_max_freq(freq, mag)
    print(f'-> {file} has ({max}) Hz')
    destination = breath_dir_path
    if max < 20:
      destination = noise_dir_path
    elif max > 50:
      destination = stimulation_dir_path
    shutil.move(file_path, os.path.join(destination, file))
