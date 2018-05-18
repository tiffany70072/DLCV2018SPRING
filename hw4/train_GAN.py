from import_data import import_image
from model import GAN
from plot import output_32

import scipy.misc
from sklearn.metrics import mean_squared_error
import numpy as np
import sys

from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dropout
from keras.models import Model

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.15
#config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
import keras
keras.backend.set_session(sess)

class Model(object):
	def __init__(self):
		self.epochs = 11000
		self.batch_size = 128
		self.noise_size = 128
		self.sample_interval = 250 
		self.loss_interval = 10
		self.half_batch = int(self.batch_size / 2)

		self.generator, self.discriminator, self.combined = GAN()
		start_epoch = 19999
		#self.generator.load_weights('saved_model/GAN_G_epochs'+str(start_epoch)+'_weights.h5', by_name=True)
		#self.discriminator.load_weights('saved_model/GAN_D_epochs'+str(start_epoch)+'_weights.h5', by_name=True)
		#self.combined.load_weights('saved_model/GAN_C_epochs'+str(start_epoch)+'_weights.h5', by_name=True)

	def output(self, pred, name):
		print("pred =", pred.shape)
		for i in range(pred.shape[0]):
			scipy.misc.imsave("output/" + name + "_" + str(i) + '.png', pred[i])

	def train_D(self):
		idx = np.random.randint(0, self.x_train.shape[0], self.half_batch)
		imgs = self.x_train[idx]
		noise = np.random.normal(0, 1, (self.half_batch, self.noise_size))
		# Generate a half batch of new images
		gen_imgs = self.generator.predict(noise)

		# Train the discriminator
		d_loss_real = self.discriminator.train_on_batch(imgs, np.ones((self.half_batch, 1)))
		d_loss_fake = self.discriminator.train_on_batch(gen_imgs, np.zeros((self.half_batch, 1)))
		self.d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

	def train_G(self):
		noise = np.random.normal(0, 1, (self.batch_size, self.noise_size))
		valid_y = np.array([1] * self.batch_size)
		self.g_loss = self.combined.train_on_batch(noise, valid_y)

	def train_GAN(self):
		#self.x_train = import_image('test')
		self.x_train = import_image('train')
		#x_test  = import_image('test')
		self.x_train = self.x_train * 2 - 1
		print("x_train =", self.x_train.shape)

		self.half_batch = int(self.batch_size / 2)
		print('construct noise')
		noise = np.random.normal(0, 1, (32, 128))
		np.save('GAN_noise.npy', noise)
		history = []

		self.train_D()
		self.train_G()
		fout = open('process2', 'a')
		for epoch in range(9000, 9000+self.epochs):
			if self.d_loss[1] > 0.8: 
				print("%d" %1, end = ' ')
				fout.write("%d " %1)
				for i in range(15): self.train_G()
			elif self.d_loss[1] > 0.65:
				print("%d" %2, end = ' ')
				fout.write("%d " %2)
				for i in range(5): self.train_G()
					
			elif self.d_loss[1] < 0.55: 
				print("%d" %4, end = ' ') 
				fout.write("%d " %4)
				for i in range(2): self.train_D()
			elif self.d_loss[1] < 0.4: 
				print("%d" %5, end = ' ') 
				fout.write("%d " %5)
				for i in range(7): self.train_D()
	
			else:
				print("%d" %3, end = ' ') 
				fout.write("%d " %3)
			self.train_D()
			self.train_G()
			# Plot the progress
			if epoch % 10 == 0:
				print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, self.d_loss[0], 100*self.d_loss[1], self.g_loss))
				fout.write("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]\n" % (epoch, self.d_loss[0], 100*self.d_loss[1], self.g_loss))
			if epoch % self.loss_interval == 0:
				history.append([self.d_loss[0], 100*self.d_loss[1], self.g_loss])

			# If at save interval => save generated image samples
			if epoch % self.sample_interval == 0:
				self.sample_images(epoch, noise)
			if epoch % 500 == 0:
				self.generator.save_weights('saved_model/GAN_G_epochs' + str(epoch) +'_weights.h5')
				self.discriminator.save_weights('saved_model/GAN_D_epochs' + str(epoch) +'_weights.h5')
				self.combined.save_weights('saved_model/GAN_C_epochs' + str(epoch) +'_weights.h5')
				np.save("GAN_history2", np.array(history))
		history.append([self.d_loss[0], 100*self.d_loss[1], self.g_loss])
		self.generator.save_weights('saved_model/GAN_G_epochs' + str(epoch) +'_weights.h5')
		self.discriminator.save_weights('saved_model/GAN_D_epochs' + str(epoch) +'_weights.h5')
		self.combined.save_weights('saved_model/GAN_C_epochs' + str(epoch) +'_weights.h5')
		np.save("GAN_history2", np.array(history))
		self.sample_images(20000, noise)

	def sample_images(self, epoch, noise):
		print('sample...')
		gen_imgs = self.generator.predict(noise)
		gen_imgs = 0.5 * gen_imgs + 0.5
		print("gen_imgs =", gen_imgs.shape)
		#self.output(gen_imgs, str(epoch))
		output_32(gen_imgs, str(epoch), "output/random_")

	def fixed_seed(self):
		seed = np.array([2, 3, 4, 8, 13, 15, 16])
		position = np.array([3, 16, 4, 13, 4, 4, 8, 8, 8, 3, 13, 4, 2, 13, 13, 8, \
							2, 8, 2, 3, 3, 13, 16, 8, 4, 15, 3, 2, 4, 8, 4, 13])
		noise = np.empty((32, 128))
		count = []
		for i in range(seed.shape[0]):
			np.random.seed(seed[i])
			temp = np.random.normal(0, 1, (32, 128))
			print(seed[i])
			for j in range(32): 
				if position[j] == seed[i]: noise[j] = temp[j]  
			#count.append(j)
		#print(count, len(count))
		return noise

	def plot_random(self):
		self.generator.load_weights('saved_model/GAN_G_epochs19999_weights.h5', by_name=True)
		#self.discriminator.load_weights('saved_model/GAN_D_epochs'+str(start_epoch)+'_weights.h5', by_name=True)
		#self.combined.load_weights('saved_model/GAN_C_epochs'+str(start_epoch)+'_weights.h5', by_name=True)
		noise = self.fixed_seed()
		gen_imgs = self.generator.predict(noise)
		gen_imgs = 0.5 * gen_imgs + 0.5
		output_32(gen_imgs, sys.argv[2] + "fig2_3.jpg", 'trash')
		#output_32(gen_imgs, sys.argv[2] + "fig2_3.jpg", str(i))
		
def main():
	gan = Model()
	#gan.train_GAN()
	gan.plot_random()
	
if __name__ == '__main__':
	main()