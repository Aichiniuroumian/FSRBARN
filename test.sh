# helen
python test.py --gpus 1 --model Ours --name Ours \
    --load_size 128 --dataset_name single --dataroot /home/wang107552002794/Ours/test_dirs/Helen_test/LR\
    --pretrain_model_path /home/wang107552002794/Ours/pretrain_models/Ours.pth\
    --save_as_dir results_helen/Ours_test/
# celeba
python test.py --gpus 1 --model Ours --name Ours \
    --load_size 128 --dataset_name single --dataroot /home/wang107552002794/Ours/test_dirs/CelebA_test_DIC/LR \
    --pretrain_model_path /home/wang107552002794/Ours/pretrain_models/Ours.pth\
    --save_as_dir results_CelebA/Ours_test/

# ----------------- calculate PSNR/SSIM scores ----------------------------------
python psnr_ssim.py
# ------------------------------------------------------------------------------- 

