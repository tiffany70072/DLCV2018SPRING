#!/bin/bash

mkdir saved_model
echo p2 testing
wget -O saved_model/p2_rnn_weights.h5 "dl=1"
time CUDA_VISIBLE_DEVICES=0 python3 src/trimmed_action_recognition.py $1 $2 $3