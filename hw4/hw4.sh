#!/bin/bash

mkdir saved_model

echo VAE testing

CUDA_VISIBLE_DEVICES=0 python3 train_VAE.py $1 $2
echo plot loss
python3 plot_loss.py VAE $2
echo tsne
python3 tsne.py $1 $2

echo
echo GAN
wget -O saved_model/GAN_C_epochs19999_weights.h5 "https://www.dropbox.com/s/u4s2mto233lztnd/GAN_C_epochs19999_weights.h5?dl=1"
wget -O saved_model/GAN_D_epochs19999_weights.h5 "https://www.dropbox.com/s/8sd4yk21qv4tljm/GAN_D_epochs19999_weights.h5?dl=1"
wget -O saved_model/GAN_G_epochs19999_weights.h5 "https://www.dropbox.com/s/3x6s99ee227uu83/GAN_G_epochs19999_weights.h5?dl=1"

CUDA_VISIBLE_DEVICES=0 python3 train_GAN.py $1 $2
echo plot loss 
python3 plot_loss.py GAN $2

echo ACGAN

wget -O saved_model/ACGAN_C_epochs20000_weights.h5 "https://www.dropbox.com/s/edl6oa3jj606jn6/ACGAN_C_epochs20000_weights.h5?dl=1"
wget -O saved_model/ACGAN_D_epochs20000_weights.h5 "https://www.dropbox.com/s/21ckekjrmy7vjge/ACGAN_D_epochs20000_weights.h5?dl=1"
wget -O saved_model/ACGAN_G_epochs20000_weights.h5 "https://www.dropbox.com/s/7jjppwdfgff97xd/ACGAN_G_epochs20000_weights.h5?dl=1"

CUDA_VISIBLE_DEVICES=0 python3 train_ACGAN.py $1 $2
echo plot loss
python3 plot_loss.py ACGAN $2