import os

from os import listdir
from os.path import isfile, join

def only_files(dir_path: str):
  return [f for f in listdir(dir_path) if isfile(join(dir_path, f)) and f != '.DS_Store']

def only_folders(dir_path: str):
  return [f for f in listdir(dir_path) if isfile(join(dir_path, f)) is False and f != '.DS_Store']

def create_folder(path: str):
  if not os.path.exists(path):
    os.makedirs(path)

def combine_audio(self, in_path: str, out_path: str, fps=25):
  vide_clip = VideoFileClip(f'{in_path}.mkv')
  audio_background = AudioFileClip(f'{in_path}.wav')
  vide_clip.audio = audio_background
  vide_clip.write_videofile(f'{out_path}.mkv', fps=fps)