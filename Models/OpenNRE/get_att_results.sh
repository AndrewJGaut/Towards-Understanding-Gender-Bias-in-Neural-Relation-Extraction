

gpu=0

CUDA_VISIBLE_DEVICES=$gpu nohup python3 -u train_debiasingoptions.py > reg.txt
wait

CUDA_VISIBLE_DEVICES=$gpu nohup python3 -u train_debiasingoptions.py -gs > gs.txt &
wait

CUDA_VISIBLE_DEVICES=$gpu nohup python3 -u train_debiasingoptions.py -na > na.txt &
wait


