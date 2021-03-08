data_path=~/DATASET/KITTI/data

# train with pretrained model
# CUDA_VISIBLE_DEVICES=3 python train.py --model_name M_640x192\
#   --height 192 --width 640 --data_path $data_path

# train w/o pretrain
# CUDA_VISIBLE_DEVICES=0 python train.py --model_name M_no_pt_640x192\
#   --height 192 --width 640 --data_path $data_path --weights_init scratch

# train with pretrained model and attention
CUDA_VISIBLE_DEVICES=1 python train.py --model_name M_att_640x192_b24\
 --height 192 --width 640 --data_path $data_path --enable_attention 1\
 --batch_size 24

# train with pretrained model and attention with higher resolution
# CUDA_VISIBLE_DEVICES=0 python train.py --model_name M_att_1024x320\
#   --height 320 --width 1024 --data_path ~/DATASET/KITTI/data --enable_attention 1

