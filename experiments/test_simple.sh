model_folder='/home/yida/Desktop/depth-estimation/model'
atten_folder='/home/yida/Desktop/depth-estimation/0926_0061_sync_02_atten'
src_folder='/home/yida/Desktop/depth-estimation/0926_0061_sync_02_source'
python atten_test.py --image_path $atten_folder --model_name $model_folder
python test_simple.py --image_path $src_folder --model_name 'mono+stereo_1024x320'