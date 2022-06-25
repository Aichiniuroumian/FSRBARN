# Face-Super-resolution-Reconstruction-Based-on-Attention-Residual-Network
Face Super-resolution Reconstruction Based on  Attention Residual Network
I have tested the codes on

- Ubuntu 18.04
- CUDA 10.1  
- Python 3.7, install required packages by `pip3 install -r requirements.txt`  

Test with Pretrained Models

Pre-training model download address：[pretrain_models](https://pan.baidu.com/s/1rtYgHx-jQtiD00HBVPg4qQ) 
Password：scsb

```
# helen
python test.py --gpus 1 --model Ours --name Ours \    --load_size 128 --dataset_name single --dataroot /home/wang107552002794/Ours/test_dirs/Helen_test/LR\    --pretrain_model_path /home/wang107552002794/Ours/pretrain_models/Ours.pth\    --save_as_dir results_helen/Ours_test/


# celeba
python test.py --gpus 1 --model Ours --name Ours \    --load_size 128 --dataset_name single --dataroot /home/wang107552002794/Ours/test_dirs/CelebA_test/LR \  --pretrain_model_path /home/wang107552002794/Ours/pretrain_models/Ours.pth\    --save_as_dir results_CelebA/Ours_test/

```

### Train the Model

The commands used to train the released models are provided in script `train.sh`. Here are some train tips:

- You should download [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)  train Ours. 
- To train Ours, we simply crop out faces from CelebA without pre-alignment, because for ultra low resolution face SR, it is difficult to pre-align the LR images.  
- Please change the `--name` option for different experiments. Tensorboard records with the same name will be moved to `check_points/log_archive`, and the weight directory will only store weight history of latest experiment with the same name.  

**Acknowledgement**

The codes are based on [SPARNet](https://github.com/chaofengc/Face-SPARNet) . The project also benefits from [DICNet](https://github.com/Maclory/Deep-Iterative-Collaboration).  
