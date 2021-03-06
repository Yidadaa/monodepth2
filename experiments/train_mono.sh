data_path=~/DATASET/KITTI/data

# train with pretrained model
# CUDA_VISIBLE_DEVICES=1 python train.py --model_name M_pt_raw_640x192\
#   --height 192 --width 640 --data_path $data_path

# train w/o pretrain
# CUDA_VISIBLE_DEVICES=0 python train.py --model_name M_no_pt_640x192\
#   --height 192 --width 640 --data_path $data_path --weights_init scratch

# train with pretrained model and attention[last layer]
# CUDA_VISIBLE_DEVICES=1 python train.py --model_name M_atten_3_640x192\
#   --height 192 --width 640 --data_path $data_path --atten_layer 3

# train with pretrain and with pose consistency
# CUDA_VISIBLE_DEVICES=0 python train.py --model_name M_pose_l2_640x192\
#    --height 192 --width 640 --data_path $data_path\
#    --atten_layer -1 --use_pose_consistency 1

# train with pretrain and with pose consistency
# CUDA_VISIBLE_DEVICES=2 python train.py --model_name test\
#   --height 192 --width 640 --data_path $data_path\
#   --atten_layer -1 --batch_size 2 --use_pose_consistency 1

# train with pretrain and with pose consistency
# CUDA_VISIBLE_DEVICES=1 python train.py --model_name M_att_fuse_4_640x192\
#    --height 192 --width 640 --data_path $data_path\
#    --atten_layer 4 --use_pose_consistency 0

# train with pretrain and with pose consistency projection
# CUDA_VISIBLE_DEVICES=2 python train.py --model_name M_pose_fb_ln_norm_640x192\
#    --height 192 --width 640 --data_path $data_path\
#    --use_pose_cons_proj 0 --use_pose_consistency 1

# train with pretrain and with encode_pos
# CUDA_VISIBLE_DEVICES=1 python train.py --model_name M_pose_position_center_offet_640x192\
#   --height 192 --width 640 --data_path $data_path\
#   --use_pose_cons_proj 0 --use_pose_consistency 0\
#   --encode_pos 1

# train with pretrained model disable auto mask
CUDA_VISIBLE_DEVICES=1 python train.py --model_name M_pt_disable_automask_640x192\
  --height 192 --width 640 --data_path $data_path --disable_automasking


