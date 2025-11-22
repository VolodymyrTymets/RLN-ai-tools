import os
import numpy as np
from os import listdir
from os.path import isfile, join
import shutil

ASSETS_FOLDER = 'assets'
data_set_folder = ['train', 'valid']
ratio = [0.8, 0.2]

def prepare_folder(folder_name: str, label: str):
  folder_path = os.path.join(ASSETS_FOLDER, folder_name, label)
  only_files = [f for f in listdir(folder_path) if isfile(join(folder_path, f)) and f != '.DS_Store']
  count_files = len(only_files)
  valid_count = int(count_files * ratio[1])
  train_count = count_files - valid_count
  np.random.shuffle(only_files)
  # train_files = only_files[:train_count]
  # valid_files = only_files[train_count:]

  for group in ['valid', 'train']:
    group_path = os.path.join(ASSETS_FOLDER, folder_name, group)
    if not os.path.exists(group_path):
      os.makedirs(group_path)

    for file in only_files[train_count:] if group == 'valid' else only_files[:train_count]:
      from_path = os.path.join(folder_path, file)
      to_path = os.path.join(ASSETS_FOLDER, folder_name, group, label)
      print('--> move:', from_path, 'to:', to_path)
      if not os.path.exists(to_path):
        os.makedirs(to_path)
      shutil.move(from_path, os.path.join(to_path, file))

def main():
  for folder in ['noise', 'breath', 'stimulation']:
    print('Prepare Folder:', folder)
    prepare_folder('data_set_2000', folder)


if __name__ == '__main__':
    main()