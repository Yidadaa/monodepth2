CUDA_VISIBLE_DEVICES=1 python evaluate_depth.py\
  --load_weights_folder ~/tmp/M_att_640x192-21_02_24-00:26:36/models/weights_17\
  --eval_mono --data_path ~/DATASET/KITTI/data --enable_attention 1