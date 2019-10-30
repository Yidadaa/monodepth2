'''Check if the files in the split_files have been downloaded.
'''

import os
import argparse
from tqdm import tqdm

def parse():
  parser = argparse.ArgumentParser(description="Parse mono options.")
  parser.add_argument("--data_path", type=str, help="path to the training data")
  parser.add_argument("--check_path", type=str, help="split files folder")
  return parser.parse_args()

def get_image_path(data_path:str, folder:str, frame_index:int, side:str):
  side_map = {"2": 2, "3": 3, "l": 2, "r": 3}
  f_str = "{:010d}{}".format(int(frame_index), '.jpg')
  image_path = os.path.join(
    data_path, folder, "image_0{}/data".format(side_map[side]), f_str)
  return image_path

def main():
  args = parse()

  data_path, check_path = args.data_path, args.check_path
  file_lists = []

  for fname in os.listdir(check_path):
    with open(os.path.join(check_path, fname), 'r') as f:
      file_lists += f.read().split('\n')

  missed_files = []
  missed_folders = set()
  for file in tqdm(file_lists, desc='Checking'):
    if len(file.split(' ')) < 3: continue
    folder, frame_index, side = file.split(' ')
    file_path = get_image_path(data_path, folder, frame_index, side)
    if not os.path.exists(file_path):
      missed_files.append(file_path)
      if folder not in missed_folders:
        missed_folders.add(folder)
  
  print('Missed files: {}'.format(len(missed_files)))
  print('Missed folders: {}'.format(len(missed_folders)))
  print(missed_folders)


if __name__ == "__main__":
  main()