#!/bin/bash

mkdir saved_model
echo p1 testing
wget -O saved_model/p1_cnn_weights.h5 "dl=1"
time CUDA_VISIBLE_DEVICES=0 python3 src/data_preprocessing.py $1 $2 $3