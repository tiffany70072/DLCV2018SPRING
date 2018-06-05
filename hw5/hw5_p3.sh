#!/bin/bash

mkdir saved_model
echo p3 testing
wget -O saved_model/p3_weights.h5 "dl=1"
time CUDA_VISIBLE_DEVICES=0 python3 src/temporal_action_segmentation.py $1 $2 $3
