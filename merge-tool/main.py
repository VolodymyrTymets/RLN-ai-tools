import os
import sys
from src.utils import only_folders
from src.FragmentMergeWorker import FragmentMergeWorker


ASSETS_FOLDER = 'assets'

if __name__ == '__main__':
   target_folder = os.path.join(ASSETS_FOLDER, sys.argv[1])
   sub_folders = only_folders(target_folder)
   sub_folders = [f for f in sub_folders if '_merged' not in f]
   print('sub_folders:', sub_folders, '\n\n')
   for sub_folder in sub_folders:
     fragment_merge_worker = FragmentMergeWorker(duration=20, in_path=os.path.join(target_folder, sub_folder), out_path=os.path.join(target_folder, sub_folder + '_merged'))
     fragment_merge_worker.merge()
