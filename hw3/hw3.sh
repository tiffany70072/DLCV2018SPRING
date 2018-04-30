#!/bin/bash
mkdir hw3_model
wget -O hw3_model/VGG16FCN32_epoch40_weights.h5 "https://www.dropbox.com/s/o2ah2xlr1q9ox2x/VGG16FCN32_epoch40_weights.h5?dl=1"
CUDA_VISIBLE_DEVICES=0 python3 test_VGG16_FCN32.py $1 $2 
# hw3-train-validation/validation/ test_env/baseline/