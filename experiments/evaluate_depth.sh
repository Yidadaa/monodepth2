# CUDA_VISIBLE_DEVICES=0 python evaluate_depth.py\
#   --load_weights_folder /home/zhangyifei/tmp/M_att_640x192_b48-21_03_07-18:25:34/models/weights_17\
#   --eval_mono --data_path ~/DATASET/KITTI/data --enable_attention 1\
#   --save_pred_disps

# 最好的实验结果
# CUDA_VISIBLE_DEVICES=0 python evaluate_depth.py\
#   --load_weights_folder /home/zhangyifei/tmp/M_att_4_640x192-19_11_04-17:34:41/models/weights_19\
#   --eval_mono --data_path ~/DATASET/KITTI/data --enable_attention 1\
#   --save_pred_disps

CUDA_VISIBLE_DEVICES=0 python evaluate_depth.py\
  --load_weights_folder /home/zhangyifei/tmp/M_att_640x192_b24_self-21_03_09-22:15:33/models/weights_18\
  --eval_mono --data_path ~/DATASET/KITTI/data --enable_attention 1\
  --save_pred_disps