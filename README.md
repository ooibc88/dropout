# Customizable and Effective Dropout

![version](https://img.shields.io/badge/version-v1.0-brightgreen)
![python](https://img.shields.io/badge/python-3.7.3-blue)
![pytorch](https://img.shields.io/badge/pytorch-1.2.0-informational)

This repository contains code for the paper 
[Effective and Efficient Dropout for Deep Convolutional Neural Networks](https://arxiv.org/abs/1904.03392).
Customizable and effective Dropout blocks have been proposed to support complex analytics with Convolutional Neural Networks.

The illustration of convolutional transformations with 4 structural levels of dropout: 
1. Dropout, or drop-neuron, gates input neurons in operation 1;
2. Drop-channel replaces identity mapping in operation 2 with operation 3, random sampling and gating on channels; 
3. Drop-path is introduced to F conv in operation 4;
4. Drop-layer to the shortcut connection in operation 5.

<img src="https://user-images.githubusercontent.com/14588544/70496794-f11e1880-1b4b-11ea-83a5-381931a0c2f6.png" width=78%/>

The illustration of the example proposed building block:

<img src="https://user-images.githubusercontent.com/14588544/70497240-78b85700-1b4d-11ea-9119-12e5dfdf4012.png" width=30%/>


### The repo includes:

1. example models (/models)
2. codes for dropout training (train.py)
3. codes to support different structural levels of dropout (models/convBlock.py)
    * supporting effective dropout with customizable building blocks (models/convBlock/conv_block)

### Training
##### Dependencies
    * python 3.7.3
    * pytorch 1.2.0
    * torchvision 0.4.0
##### Model Training

```
Example training code:
CUDA_VISIBLE_DEVICES=0 python train.py --net_type=resnet --depth 110 --arg1 1 --epoch 164 --weight_decay 1e-4 --block_type 0 --drop_type=1 --drop_rate=0.1 --exp_name resnet_dropChannel --report_ratio

Please check help info in argparse.ArgumentParser (train.py) for more details 
```


### Contact
To ask questions or report issues, please open an issue here or can directly send [us](mailto:shaofeng@comp.nus.edu.sg) an email.