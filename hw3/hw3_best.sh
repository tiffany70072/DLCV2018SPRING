#!/bin/bash
mkdir hw3_model
wget -O hw3_model/VGG16FCN16_epoch40_weights.h5 "https://www.dropbox.com/s/9iuxutmoy9qmj43/VGG16FCN16_epoch40_weights.h5?dl=1"
CUDA_VISIBLE_DEVICES=0 python3 test_VGG16_FCN16.py $1 $2 
# hw3-train-validation/test/ test_env/best/
