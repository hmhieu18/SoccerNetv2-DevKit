cd /content/SoccerNetv2-DevKit
git reset --hard origin/master
git pull 
cd /content/SoccerNetv2-DevKit/Task1-ActionSpotting/TemporallyAwarePooling_audio_vgg
clear
#test NetVlad with vgg features
python src/main.py \
--SoccerNet_path=/content/TemporallyAwarePooling_Data/content/SoccerNet/content/TemporallyAwarePooling/SoccerNet_TemporallyAwarePooling \
--audio_path=/content/drive/MyDrive/Thesis_temp/soccernet-video \
--model_name=NetVLAD-vgg-4 \
--percent=1 \
--batch_size 32 \
--max_num_worker 4 \
--train_list="/content/splits-vgg/train_split.npy" \
--test_list="/content/splits-vgg/test_split.npy" \
--valid_list="/content/splits-vgg/valid_split.npy" \
--base_dir="/content/drive/MyDrive/Thesis_temp/training-models" \
--load_weights="/content/drive/MyDrive/Thesis_temp/training-models/models/NetVLAD-vgg-3/model.pth.tar" \
--test_only \
--load_once