data_path=~/DATASET/KITTI/data

CUDA_VISIBLE_DEVICES=3 python train.py --model_name M_640x192\
  --height 192 --width 640 --data_path $data_path