data_path=~/DATASET/KITTI/data

# train with pretrained model
# CUDA_VISIBLE_DEVICES=3 python train.py --model_name M_640x192\
#   --height 192 --width 640 --data_path $data_path

# train w/o pretrain
# CUDA_VISIBLE_DEVICES=0 python train.py --model_name M_no_pt_640x192\
#   --height 192 --width 640 --data_path $data_path --weights_init scratch

# train with pretrained model and attention[last layer]
# CUDA_VISIBLE_DEVICES=1 python train.py --model_name M_att_640x192\
#   --height 192 --width 640 --data_path $data_path --atten_layer 4

# train with pretrain and with pose consistency
CUDA_VISIBLE_DEVICES=0 python train.py --model_name M_pose_640x192\
  --height 192 --width 640 --data_path $data_path\
  --atten_layer -1 --batch_size 16 --use_pose_consistency 1