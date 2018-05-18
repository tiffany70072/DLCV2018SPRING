from import_data import import_image
from model import autoencoder_1, VAE, VAE_encoder, VAE_decoder
from plot import output_32, output_20

import scipy.misc
from sklearn.metrics import mean_squared_error
import numpy as np
import sys

from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dropout
from keras.models import Sequential
from keras.models import Model

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
sess = tf.Session(config=config)
import keras
keras.backend.set_session(sess)


def output(pred, name):
	print("pred =", pred.shape)
	for i in range(pred.shape[0]):
		scipy.misc.imsave("output_VAE2/" + name + "_" + str(i) + '.png', pred[i])

def get_MSE(real, pred):
	print("MSE =", mean_squared_error(real.reshape(-1, 64*64*3), pred.reshape(-1, 64*64*3)))

def problem3(model): # plot test image and calculate MSE
	x_test = import_image(path = sys.argv[1])
	decoded_imgs = model.predict(x_test)[0]
	#output(decoded_imgs[:10], 'test')
	pred = np.concatenate([x_test[:10], decoded_imgs], axis = 0)
	output_20(pred, sys.argv[2] + "fig1_3.jpg", 80)
	get_MSE(x_test, decoded_imgs)

def problem4(): # randomly plot
	latent_dim = 1024
	np.random.seed(3)
	z = np.random.normal(0, 1, [32, latent_dim])
	z_tensor = Input(shape=(latent_dim, ))
	print("z =", z.shape)

	decoder_output = VAE_decoder(z_tensor)
	decoder = Model(z_tensor, decoder_output)
	#weights_path = 'saved_model/VAE2_epochs60_3-4_weights.h5'
	weights_path = 'saved_model/VAE2_epochs_weights.h5'
	decoder.load_weights(weights_path, by_name=True)
	pred = decoder.predict(z)

	output_32(pred, sys.argv[2] + "fig1_4.jpg", 60)

def problem5():
	path = sys.argv[1]
	x_test = import_image(sys.argv[1])

	input_img = Input(shape=(64, 64, 3))
	encoder_output = VAE_encoder(input_img)
	encoder = Model(input_img, encoder_output)
	weights_path = 'saved_model/VAE2_epochs_weights.h5'
	encoder.load_weights(weights_path, by_name=True)
	h = encoder.predict(x_test)
	print("h =", h.shape)
	np.save('report/VAE2_problem5_h', h)

def train_VAE():
	model = VAE()
	model.summary()
	history = []

	x_train = import_image(folder = 'train')
	x_test  = import_image(folder = 'test')
	#x_train = import_image('test')
	arbitrary = np.zeros([x_train.shape[0], 1024*2])
	model.summary()

	history = model.fit(x_train, [x_train, arbitrary], epochs=200, batch_size=128,
		shuffle=True, verbose=1, validation_data=(x_test, [x_test, arbitrary[:x_test.shape[0]]])) 
	print("history =", history.history.keys())
	output_loss1 = history.history['recons_loss']
	output_loss2 = history.history['KLD_loss']
	np.save('VAE2_recons_loss', output_loss1)
	np.save('VAE2_KLD_loss', output_loss2)

	model.save_weights('saved_model/VAE2_epochs60_weights.h5')

def load_VAE():
	model = VAE()
	model.summary()
	model.load_weights('saved_model/VAE2_epochs_weights.h5')
	return model

def main():
	#train_VAE()

	model = load_VAE()
	problem3(model)
	problem4()
	problem5()
	
if __name__ == '__main__':
	main()