CUDA_VISIBLE_DEVICES=0 python evaluate_depth.py\
  --load_weights_folder /home/zhangyifei/tmp/M_att_640x192_b48-21_03_07-18:25:34/models/weights_17\
  --eval_mono --data_path ~/DATASET/KITTI/data --enable_attention 1
#  --save_pred_disps
