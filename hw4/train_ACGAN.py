from import_data import import_image, import_image_features
from model import ACGAN, GAN
from plot import output_20

import scipy.misc
from sklearn.metrics import mean_squared_error
import numpy as np
import sys

from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dropout
from keras.models import Model
from keras.utils import to_categorical

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.15
#config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
import keras
keras.backend.set_session(sess)

class Model(object):
	def __init__(self):
		self.epochs = 40000
		self.batch_size = 128
		self.noise_size = 128
		self.sample_interval = 100 #50
		self.loss_interval = 10
		self.half_batch = int(self.batch_size / 2)

		self.generator, self.discriminator, self.combined = ACGAN()
		self.start_epoch = 0
		#self.generator.load_weights('saved_model/GAN_G_epochs'+str(start_epoch)+'_weights.h5', by_name=True)
		#self.discriminator.load_weights('saved_model/GAN_D_epochs'+str(start_epoch)+'_weights.h5', by_name=True)
		#self.combined.load_weights('saved_model/GAN_C_epochs'+str(start_epoch)+'_weights.h5', by_name=True)

	def output(self, pred, name):
		print("pred =", pred.shape)
		for i in range(pred.shape[0]):
			scipy.misc.imsave("output_ACGAN/" + name + "_" + str(i) + '.png', pred[i])

	def train_D(self):
		idx = np.random.randint(0, self.x_train.shape[0], self.half_batch)
		imgs = self.x_train[idx]

		real_features = self.features[idx]
		gen_features = np.random.randint(0, 2, (self.half_batch, 1))
		noise = np.random.normal(0, 1, (self.half_batch, self.noise_size))
		noise = np.concatenate([noise, gen_features], axis = 1)
		# Generate a half batch of new images
		gen_imgs = self.generator.predict(noise)
		ones = np.ones((self.half_batch, 1))
		zeros = np.zeros((self.half_batch, 1))
		real_label = np.concatenate([ones, real_features], axis = 1)
		fake_label = np.concatenate([zeros, gen_features], axis = 1)

		# Train the discriminator
		d_loss_real = self.discriminator.train_on_batch(imgs, [ones, real_features]) #
		d_loss_fake = self.discriminator.train_on_batch(gen_imgs, [zeros, gen_features]) #[np.zeros((self.half_batch, 1)), gen_features]
		self.d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

	def train_G(self):
		noise = np.random.normal(0, 1, (self.batch_size, self.noise_size))
		features = np.random.randint(0, 2, (self.batch_size, 1))

		ones = np.ones((self.batch_size, 1))
		valid_y = np.concatenate([ones, features], axis = 1)
		
		noise = np.concatenate([noise, features], axis = 1)
		self.g_loss = self.combined.train_on_batch(noise, [ones, features]) #[valid_y, features_cat]

	def train_ACGAN(self):
		self.x_train = import_image('train')
		self.features = import_image_features('train')
		#self.features_cat = to_categorical(self.features, 2)
		print("feat =", self.features.shape, self.features[:5])
		
		self.x_train = self.x_train * 2 - 1
		print("x_train =", self.x_train.shape)

		self.half_batch = int(self.batch_size / 2)
		noise = np.random.normal(0, 1, (10, 128))
		history = []

		self.train_D()
		self.train_G()
		fout = open('process_ACGAN', 'w')
		for epoch in range(self.epochs):
			if self.d_loss[3] > 0.8: #g_loss[1] > 3: 
				print("%d" %1, end = ' ')
				fout.write("%d " %1)
				for i in range(15): self.train_G()
			elif self.d_loss[3] > 0.65: #g_loss[1] > 1.5:
				print("%d" %2, end = ' ')
				fout.write("%d " %2)
				for i in range(5): self.train_G()
					
			elif self.d_loss[3] < 0.45: #[1] > 1.8: 
				print("%d" %4, end = ' ') 
				fout.write("%d " %4)
				for i in range(2): self.train_D()
			else:
				print("%d" %3, end = ' ') 
				fout.write("%d " %3)
			self.train_D()
			self.train_G()
			'''
			if self.g_loss[1] > 2: #g_loss[1] > 3: 
				print("%d" %1, end = ' ')
				fout.write("%d " %1)
				self.train_G()
			elif self.g_loss[1] > 1.5: #g_loss[1] > 1.5:
				print("%d" %2, end = ' ')
				fout.write("%d " %2)
				for i in range(5):
					self.train_G()
				self.train_D()
			elif self.d_loss[1] > 2: #g_loss[1] > 1.5:
				print("%d" %4, end = ' ')
				fout.write("%d " %4)
				self.train_D()
			else:
				print("%d" %3, end = ' ') 
				fout.write("%d " %3)
				self.train_D()
				self.train_G()
			'''
			# Plot the progress
			if epoch % 10 == 0:
				print("[acc1.: %.2f%%, acc2.: %.2f%%]" % (100*self.d_loss[3], 100*self.d_loss[4])) # 3, 4
				print("%d [D loss1: %.4f, D loss2.: %.4f] [G loss1: %.4f, G loss2: %.4f]" % \
					(epoch, self.d_loss[1], self.d_loss[2], self.g_loss[1], self.g_loss[2])), # 1, 2
				fout.write("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]\n" % (epoch, self.d_loss[0], 100*self.d_loss[3], self.g_loss[0]))

			if epoch % self.loss_interval == 0:
				history.append([self.d_loss[1], self.d_loss[2], self.g_loss[1], \
					self.g_loss[2], 100*self.d_loss[3], 100*self.d_loss[4]])

			# If at save interval => save generated image samples
			if epoch % self.sample_interval == 0:
				self.sample_images(epoch, noise)
			if epoch % 1000 == 0:
				self.generator.save_weights('saved_model/ACGAN_G_epochs' + str(epoch) +'_weights.h5')
				self.discriminator.save_weights('saved_model/ACGAN_D_epochs' + str(epoch) +'_weights.h5')
				self.combined.save_weights('saved_model/ACGAN_C_epochs' + str(epoch) +'_weights.h5')
				np.save("ACGAN_history", np.array(history))
		#history.append([self.d_loss[0], 100*self.d_loss[1], self.g_loss])
		self.generator.save_weights('saved_model/GAN_G_epochs' + str(epoch) +'_weights.h5')
		self.discriminator.save_weights('saved_model/GAN_D_epochs' + str(epoch) +'_weights.h5')
		self.combined.save_weights('saved_model/GAN_C_epochs' + str(epoch) +'_weights.h5')
		np.save("ACGAN_history", np.array(history))
		self.sample_images(epoch, noise)

	def sample_images(self, epoch, noise):
		#generator, _, _ = GAN()
		noise = np.concatenate([noise, noise], axis = 0)
		ones  = np.ones((10, 1))
		zeros = np.zeros((10, 1))
		
		feature = np.concatenate([ones, zeros], axis = 0)
		noise = np.concatenate([noise, feature], axis = 1)

		gen_imgs = self.generator.predict(noise)
		gen_imgs = 0.5 * gen_imgs + 0.5
		print("gen_imgs =", gen_imgs.shape)
		#output_20(gen_imgs, "output_ACGAN/ACGAN_", str(epoch))
		output_20(gen_imgs, sys.argv[2] + 'fig3_3.jpg', str(epoch))

	def plot(self):
		np.random.seed(8)
		noise = np.random.normal(0, 1, (10, 128))
		self.generator.load_weights('saved_model/ACGAN_G_epochs20000_weights.h5', by_name=True)
		self.sample_images(8, noise)
	
def main():
	gan = Model()
	#gan.train_ACGAN()
	gan.plot()
	
if __name__ == '__main__':
	main()