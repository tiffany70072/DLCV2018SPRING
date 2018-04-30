from load_data import import_image, import_test_image
import tensorflow as tf
import numpy as np

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
#config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
import keras
keras.backend.set_session(sess)

import sys
import scipy.misc

from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dropout
from keras.layers import Activation, ZeroPadding2D, Cropping2D
from keras.layers.convolutional import Conv2DTranspose
from keras.models import Model

from train_FCN16 import VGG16_FCN16
from train_FCN32 import VGG16_FCN32

import os
from skimage import io

def import_image(fileIds): # import a specific image
	print("Import visualization image...")
	path = "hw3-train-validation/validation/"
	n = len(fileIds)
	sat  = np.empty([n, 512, 512, 3])
	for i in range(n):
		img = (io.imread(path+fileIds[i]+"_sat.jpg"))
		sat[i] = img/255.0

	print("sat =", sat.shape)
	return sat

def import_validation_image(): # import all images in validation set
	print("Import image...")
	path = "hw3-train-validation/validation/"
	num = 257
	sat  = np.empty([num, 512, 512, 3])
	count = 0
	for filename in os.listdir(path):
		count += 1			
		img = (io.imread(path+filename))
		idx = int(filename[:4])
		if filename[-3:] == 'jpg': 	 sat[idx] = img	/255.0	
		if count % 100 == 0: print(count)
	return sat

def output(pred, idx):
	print(pred[0][0][:10])
	pred = np.argmax(pred, axis=-1)
	print("pred =", pred.shape)
	print(pred[0][0][:10])
	pred_output = np.empty([pred.shape[0], 512, 512, 3])
	for i in range(pred.shape[0]):
		pred_output[i, pred[i] == 0] = [0, 0, 0]
		pred_output[i, pred[i] == 1] = [0, 0, 1]
		pred_output[i, pred[i] == 2] = [0, 1, 0]
		pred_output[i, pred[i] == 3] = [0, 1, 1]
		pred_output[i, pred[i] == 4] = [1, 0, 1]
		pred_output[i, pred[i] == 5] = [1, 1, 0]
		pred_output[i, pred[i] == 6] = [1, 1, 1]
		scipy.misc.imsave("report/" + idx[i] + '_mask_1.png', pred_output[i])

def output_one(pred, idx, path):
	#if idx % 10 == 0: print(pred[:10])
	pred = np.argmax(pred, axis=-1)
	#if idx % 10 == 0: print("pred =", pred.shape)
	#if idx % 10 == 0: print(pred[:10])
	pred_output = np.empty([512, 512, 3])
	pred = pred.reshape(512, 512)
	
	pred_output[pred == 0] = [0, 1, 1]
	pred_output[pred == 1] = [1, 1, 0]
	pred_output[pred == 2] = [1, 0, 1]
	pred_output[pred == 3] = [0, 1, 0]
	pred_output[pred == 4] = [0, 0, 1]
	pred_output[pred == 5] = [1, 1, 1]
	pred_output[pred == 6] = [0, 0, 0]

	scipy.misc.imsave(path + str(idx).zfill(4) + '_mask_fcn16_1.png', pred_output)

def test_import_images(test_dir):
	print("Import testing images...")
	file_list = [file for file in os.listdir(test_dir) if file.endswith('.jpg')]
	file_list.sort()
	num = len(file_list)
	print("Num =", num)
	sat = np.empty([num, 512, 512, 3], dtype = np.float32)
	idx_list = np.empty([num], dtype = np.int16)
	for i, file in enumerate(file_list):
		img = (io.imread(test_dir+file))
		idx = int(file[:4])
		sat[i] = img/255.0	
		idx_list[i] = idx
	return sat, idx_list

def test_output_one(pred, idx, output_dir):
	pred = np.argmax(pred, axis=-1)
	pred_output = np.empty([512, 512, 3])
	pred = pred.reshape(512, 512)
	
	pred_output[pred == 0] = [0, 1, 1]
	pred_output[pred == 1] = [1, 1, 0]
	pred_output[pred == 2] = [1, 0, 1]
	pred_output[pred == 3] = [0, 1, 0]
	pred_output[pred == 4] = [0, 0, 1]
	pred_output[pred == 5] = [1, 1, 1]
	pred_output[pred == 6] = [0, 0, 0]

	scipy.misc.imsave(output_dir + str(idx).zfill(4) + '_mask.png', pred_output)

def main():
	fileIds = ["0008", "0097", "0107", "0000", "0001", "0002", "0003", "0004", "0005", 
				"0006", "0007", "0009", "0010", "0011", "0012", "0013", "0014", "0015"]
	fileIds = ["0008", "0097", "0107"] 
	sat_vis = import_image(fileIds)
	#sat_val = import_validation_image()
	#model = VGG16_FCN32('hw3_model/n2048_epoch40_weights.h5')
	model = VGG16_FCN16('hw3_model/f16_epoch1_weights.h5')
	model.summary()
	'''
	for i in range(sat_val.shape[0]):
		pred_val = model.predict(sat_val[i:i+1])
		output_one(pred_val[0], i, "output/")
	'''

	pred_vis = model.predict(sat_vis)
	print("pred =", pred_vis.shape)
	for i in range(3):
		output_one(pred_vis[i], fileIds[i], "report/")

if __name__ == '__main__':
	main()