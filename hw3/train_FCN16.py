from load_data import import_image, import_test_image
import tensorflow as tf
import numpy as np

config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.2
#config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
import keras
keras.backend.set_session(sess)
from sklearn.utils import shuffle

import sys
import scipy.misc

from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dropout
from keras.layers import Activation, ZeroPadding2D, Cropping2D, Reshape, add
from keras.layers.convolutional import Conv2DTranspose
from keras.models import Model
from keras.optimizers import Adam

def VGG16_FCN16(weights_path):
	print("build_model...")
	
	# Build conv layers and pooling layers according to VGG16 models
	img_input = Input(shape=(512, 512, 3))

	# convolutional part
	x = ZeroPadding2D(padding=(100, 100), data_format=None)(img_input)
	x = Conv2D(64, (3, 3), activation='relu', padding='valid', name='block1_conv1')(x)
	x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
	print("1", x.shape)
	x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
	x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
	print("2", x.shape)
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
	print("3", x.shape)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
	pool4 = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
	print("4", x.shape)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(pool4)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
	print("5", x.shape)

	n6 = 2048
	x = Conv2D(n6, (7, 7), activation='relu', padding='valid', name='FCN_block6_conv1')(x)
	x = Dropout(0.5)(x)
	print("FCN1", x.shape)
	x = Conv2D(n6, (1, 1), activation='relu', padding='valid', name='FCN_block6_conv2')(x)
	x = Dropout(0.5)(x)
	print("FCN2", x.shape)
	x = Conv2D(7, (1, 1), padding='valid', name='FCN_block6_conv3')(x)
	print("FCN3", x.shape)
	
	upscore2 = Conv2DTranspose(7, (4, 4), strides=(2, 2), padding='valid', 
		name='FCN_block7', use_bias = False)(x)
	print("upscore2", upscore2.shape)
	score_pool4 = Conv2D(7, (1, 1), strides=(1, 1), padding='valid')(pool4)
	print("score_pool4", score_pool4.shape)
	c_size = 5
	print(score_pool4.shape, c_size)
	score_pool4c = Cropping2D(cropping=((c_size, c_size), (c_size, c_size)))(score_pool4)
	print("score_pool4c", score_pool4c.shape)
	fuse_pool4 = add([upscore2, score_pool4c])
	print("fuse_pool4", fuse_pool4.shape)
	
	upscore16 = Conv2DTranspose(7, (32, 32), strides=(16, 16), padding='valid', 
		use_bias=False)(fuse_pool4)
	print("upscore16", upscore16.shape)
	c_size = 24
	x = Cropping2D(cropping=((c_size, c_size), (c_size, c_size)))(upscore16)
	print("x", x.shape)
	
	x = Reshape((512*512, 7))(x)
	img_output = Activation('softmax')(x)
	print("Output", x.shape)

	model = Model(img_input, img_output)
	print("load weight from", weights_path)
	# Load pretrained weights according to layer names
	model.load_weights(weights_path, by_name=True)
	
	return model

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
		scipy.misc.imsave(sys.argv[2] + idx[i] + '_mask.png', pred_output[i])
	np.save("output/test", pred[0])

def main():
	#sat_train, mask_train = import_image('validation')
	sat_train, mask_train = import_image('train')
	sat_valid, mask_valid = import_image('validation')
	sat_train, mask_train = shuffle(sat_train, mask_train, random_state=0)
	#mask_train = np.reshape(mask_train, (-1, 512*512, 7))
	#mask_valid = np.reshape(mask_valid, (-1, 512*512, 7))
	
	#model = VGG16_FCN16('vgg16_weights_tf_dim_ordering_tf_kernels.h5')
	model = VGG16_FCN16('hw3_model/f16_epoch30_weights.h5')
	model.summary()
	opt = Adam(lr = 1e-4) # 'adam'
	model.compile(loss = 'categorical_crossentropy', optimizer = opt, metrics = ['accuracy'])
	
	'''model.fit(x = sat_train, y = mask_train, batch_size = 2, epochs = 1, shuffle = True,
		validation_data = (sat_valid, mask_valid))
	model.save_weights('hw3_model/f16_epoch1_weights.h5')

	model.fit(x = sat_train, y = mask_train, batch_size = 2, epochs = 1, shuffle = True,
		validation_data = (sat_valid, mask_valid), verbose = 1)
	model.save_weights('hw3_model/f16_epoch2_weights.h5')

	model.fit(x = sat_train, y = mask_train, batch_size = 2, epochs = 8, shuffle = True,
		validation_data = (sat_valid, mask_valid), verbose = 1)
	model.save_weights('hw3_model/f16_epoch10_weights.h5')

	model.fit(x = sat_train, y = mask_train, batch_size = 2, epochs = 10, shuffle = True,
		validation_data = (sat_valid, mask_valid), verbose = 1)
	model.save_weights('hw3_model/f16_epoch20_weights.h5')

	model.fit(x = sat_train, y = mask_train, batch_size = 2, epochs = 10, shuffle = True,
		validation_data = (sat_valid, mask_valid), verbose = 1)
	model.save_weights('hw3_model/f16_epoch30_weights.h5')'''

	model.fit(x = sat_train, y = mask_train, batch_size = 2, epochs = 10, shuffle = True,
		validation_data = (sat_valid, mask_valid), verbose = 1)
	model.save_weights('hw3_model/f16_epoch40_weights.h5')

if __name__ == '__main__':
	main()
