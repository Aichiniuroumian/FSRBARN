python train.py --gpus 1 --name Ours --model Ours \
    --Gnorm "bn" --lr 0.0001 --beta1 0.9 --scale_factor 8 --load_size 128 \
    --dataroot /home/wang107552002794/datasets/SR_datasets/celebA/celeba_crop_train --dataset_name celeba --batch_size 32 --total_epochs 80 \
    --visual_freq 100 --print_freq 10 --save_latest_freq 500

