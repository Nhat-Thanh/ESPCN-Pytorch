# [Pytorch] Real-Time Super-Resolution 

Implementation of ESPCN model in **Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network** paper with Pytorch. 

**Tensorflow version**: https://github.com/Nhat-Thanh/ESPCN-TF

I used Adam with optimize tuned hyperparameters instead of SGD + Momentum. 

I implemented 3 models in the paper, ESPCN-x2, ESPCN-x3, ESPCN-x4 


## Contents
- [Train](#train)
- [Test](#test)
- [Demo](#demo)
- [Evaluate](#evaluate)
- [References](#references)


## Train
You run this command to begin the training:
```
python train.py  --steps=300000             \
                 --scale=2                  \
                 --batch_size=128           \
                 --save-best-only=0         \
                 --save-every=1000          \
                 --save-log=0               \
                 --ckpt-dir="checkpoint/x2" 
```
- **--save-best-only**: if it's equal to **0**, model weights will be saved every **save-every** steps.
- **--save-log**: if it's equal to **1**, **train loss, train metric, validation loss, validation metric** will be saved every **save-every** steps.


**NOTE**: if you want to re-train a new model, you should delete all files in sub-directories in **checkpoint** directory. Your checkpoint will be saved when above command finishs and can be used for the next times, so you can train a model on Google Colab without taking care of GPU time limit.
I trained 3 models on Google Colab in 300000 steps: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nhat-Thanh/ESPCN-Pytorch/blob/main/ESPCN-Pytorch.ipynb)

You can get the models here
- [ESPCN-x2.pt](checkpoint/x2/ESPCN-x2.pt)
- [ESPCN-x3.pt](checkpoint/x3/ESPCN-x3.pt)
- [ESPCN-x4.pt](checkpoint/x4/ESPCN-x4.pt)



## Test
I use **Set5** as the test set. After Training, you can test models with scale factors **x2, x3, x4**, the result is calculated by compute average PSNR of all images.
```
python test.py --scale=2 --ckpt-path="default"
```
- **--ckpt-path="default"** means you are using default model path, aka **checkpoint/x{scale}/FSRCNN-x{scale}.h5**. If you want to use your trained model, you can pass yours to **--ckpt-path**.

## Demo 
After Training, you can test models with this command, the result is the **sr.png**.
```
python demo.py --image-path="dataset/test1.png" \
               --ckpt-path="default"            \
               --scale=2
```
- **--ckpt-path** is the same as in [Test](#test)

## Evaluate

I evaluated models with Set5, Set14, BSD100 and Urban100 dataset by PSNR:

<div align="center">

|   Model  |  Set5   |  Set14  |  BSD100  | Urban100 |
|:--------:|:-------:|:-------:|:--------:|:--------:|
| ESPCN-x2 | 38.2830 | 34.4974 |  34.3377 |  31.6791 |
| ESPCN-x3 | 34.6919 | 31.3246 |  31.3524 |     X    |
| ESPCN-x4 | 32.0646 | 29.2934 |  29.7331 |  27.0724 |


</div>

<div align="center">
  <img src="./README/test1-x234-min.png" width="1000">  
  <p><strong>Bicubic x2-x3-x4 (top), ESPCN x2-x3-x4 (bottom).</strong></p>

  <img src="./README/test2-x234-min.png" width="1000">  
  <p><strong>Bicubic x2-x3-x4 (top), ESPCN x2-x3-x4 (bottom).</strong></p>
</div>
Source: game ZingSpeed Mobile

## References
- Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network: https://arxiv.org/abs/1609.05158
- T91, General100: http://vllab.ucmerced.edu/wlai24/LapSRN/results/SR_training_datasets.zip
- Set5: https://filebox.ece.vt.edu/~jbhuang/project/selfexsr/Set5_SR.zip
- Set14: https://filebox.ece.vt.edu/~jbhuang/project/selfexsr/Set14_SR.zip
- BSD100: https://filebox.ece.vt.edu/~jbhuang/project/selfexsr/BSD100_SR.zip
- Urban100: https://filebox.ece.vt.edu/~jbhuang/project/selfexsr/Urban100_SR.zip
