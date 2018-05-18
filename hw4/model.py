import tensorflow as tf
import numpy as np

from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dropout
from keras.layers import Activation, ZeroPadding2D, BatchNormalization, Reshape
from keras.layers import Flatten, Dense, Lambda, concatenate, Conv2DTranspose
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras import backend as K
from keras import losses

# https://github.com/erilyth/DCGANs/blob/master/ACGAN-CIFAR10/acgan.py
# https://github.com/lukedeo/keras-acgan/blob/master/mnist_acgan.py
# https://github.com/keras-team/keras/blob/master/examples/mnist_acgan.py
def ACGAN2():
	cls_loss_weight = 0.0
	print("Constructing ACGAN model")
	optimizer = Adam(0.0001, 0.5)
	input_img = Input(shape=(64, 64, 3))

	#discriminator = build_discriminator_ACGAN()
	discriminator = build_discriminator_ACGAN()
	#discriminator.compile(loss=['binary_crossentropy', 'binary_crossentropy'], \
	#	optimizer=optimizer, metrics=['accuracy'], loss_weights=[1., cls_loss_weight])
	discriminator.compile(loss='binary_crossentropy', \
		optimizer=optimizer, metrics=['accuracy'])

	generator = build_generator(128)
	z = Input(shape=(128,))
	img = generator(z)
	discriminator.trainable = False
	#[valid, aux] = discriminator(img)
	valid = discriminator(img)

	#combined = Model(z, [valid, aux])
	combined = Model(z, valid)
	optimizer = Adam(0.0001, 0.5)
	combined.compile(loss='binary_crossentropy', optimizer=optimizer)
	#categorical_crossentropy

	return generator, discriminator, combined

def ACGAN():
	print("Constructing GAN model")
	
	optimizer = Adam(0.0001, 0.5) 
	input_img = Input(shape=(64, 64, 3))

	discriminator = build_discriminator_ACGAN()
	discriminator.compile(loss=['binary_crossentropy', 'binary_crossentropy'], optimizer=optimizer, metrics=['accuracy'])

	generator = build_generator(128+1)
	z = Input(shape=(128+1,))
	img = generator(z)
	discriminator.trainable = False
	[valid, aux] = discriminator(img)

	combined = Model(z, [valid, aux])
	combined.compile(loss=['binary_crossentropy', 'binary_crossentropy'], optimizer=optimizer)

	return generator, discriminator, combined

# https://github.com/eriklindernoren/Keras-GAN/blob/master/gan/gan.py
def GAN():
	print("Constructing GAN model")
	
	optimizer = Adam(0.0001, 0.5) 
	input_img = Input(shape=(64, 64, 3))

	discriminator = build_discriminator()
	discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

	generator = build_generator(128)
	z = Input(shape=(128,))
	img = generator(z)
	discriminator.trainable = False
	valid = discriminator(img)

	combined = Model(z, valid)
	combined.compile(loss='binary_crossentropy', optimizer=optimizer)

	return generator, discriminator, combined

def build_generator(noise_size):
	noise_shape = (noise_size,)
	img_shape = (64, 64, 3)
	model = Sequential()

	dim = 64
	model.add(Dense((dim*16*4*4), input_shape=noise_shape))
	model.add(Reshape((4, 4, dim*16)))
	model.add(Conv2DTranspose(dim*8, (5, 5), strides = (2, 2), activation='relu', padding='same', name='g1'))
	model.add(BatchNormalization(momentum=0.8))
	model.add(Conv2DTranspose(dim*4, (5, 5), strides = (2, 2), activation='relu', padding='same', name='g2'))
	model.add(BatchNormalization(momentum=0.8))
	model.add(Conv2DTranspose(dim*2, (5, 5), strides = (2, 2), activation='relu', padding='same', name='g3'))
	model.add(BatchNormalization(momentum=0.8))
	model.add(Conv2DTranspose(3, (5, 5), strides = (2, 2), activation='tanh', padding='same', name='recons'))
	
	model.summary()
	noise = Input(shape=noise_shape)
	img = model(noise)

	return Model(noise, img)

def build_discriminator():
	img_shape = (64, 64, 3)
	model = Sequential()

	dim = 64
	model.add(Conv2D(dim, (5, 5), strides=(2, 2), padding='same', name='d1', input_shape = img_shape))
	model.add(LeakyReLU(alpha=0.2))
	model.add(BatchNormalization(momentum=0.8))
	model.add(Conv2D(dim*2, (5, 5), strides=(2, 2), padding='same', name='d2'))
	model.add(LeakyReLU(alpha=0.2))
	model.add(BatchNormalization(momentum=0.8))
	model.add(Conv2D(dim*4, (5, 5), strides=(2, 2), padding='same', name='d3'))
	model.add(LeakyReLU(alpha=0.2))
	model.add(BatchNormalization(momentum=0.8))
	model.add(Conv2D(dim*8, (5, 5), strides=(2, 2), padding='same', name='d4'))
	model.add(LeakyReLU(alpha=0.2))
	model.add(BatchNormalization(momentum=0.8))
	model.add(Flatten())
	model.add(Dense(1, activation='sigmoid'))

	model.summary()
	img = Input(shape=img_shape)
	validity = model(img)

	return Model(img, validity)

def build_discriminator_ACGAN():
	img_shape = (64, 64, 3)
	model = Sequential()

	dim = 32
	model.add(Conv2D(dim, (5, 5), strides=(2, 2), padding='same', name='d1', input_shape = img_shape))
	model.add(LeakyReLU(alpha=0.2))
	model.add(BatchNormalization(momentum=0.8))
	model.add(Conv2D(dim*2, (5, 5), strides=(2, 2), padding='same', name='d2'))
	model.add(LeakyReLU(alpha=0.2))
	model.add(BatchNormalization(momentum=0.8))
	model.add(Conv2D(dim*4, (5, 5), strides=(2, 2), padding='same', name='d3'))
	model.add(LeakyReLU(alpha=0.2))
	model.add(BatchNormalization(momentum=0.8))
	model.add(Conv2D(dim*8, (5, 5), strides=(2, 2), padding='same', name='d4'))
	model.add(LeakyReLU(alpha=0.2))
	model.add(BatchNormalization(momentum=0.8))
	model.add(Flatten())
	#model.add(Dense(2, activation='sigmoid'))
	#model.add()

	img = Input(shape=img_shape)
	features = model(img)
	model.summary()
	validity = Dense(1, activation='sigmoid', name='generation')(features)
	aux = Dense(1, activation='sigmoid', name='auxiliary')(features)
	#validity = features
	#aux = features
	#model.add(Dense(2, activation='sigmoid'))
	#validity = model(img)
	return Model(img, [validity, aux])
	#return Model(input=img, output=features)

def VAE_encoder(input_img):
	dim = 64
	kernel_size = 5
	x = Conv2D(dim, (kernel_size, kernel_size), strides=(2, 2), activation='relu', padding='same', name='conv1')(input_img) #nb_filter, nb_row, nb_col
	#x = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(x)
	#x = BatchNormalization()(x)
	x = Conv2D(dim*2, (kernel_size, kernel_size), strides=(2, 2), activation='relu', padding='same', name='conv2')(x)
	#x = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(x)
	#x = BatchNormalization()(x)
	x = Conv2D(dim*4, (kernel_size, kernel_size), strides=(2, 2), activation='relu', padding='same', name='conv3')(x)
	#x = MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(x)
	#x = BatchNormalization()(x)
	h = Flatten()(x)
	#h = Dense(512)(x)
	print(h.shape)
	#h = Flatten()(x)
	return h

#	dim = 64
#	model.add(Dense((dim*16*4*4), input_shape=noise_shape))
#	model.add(Reshape((4, 4, dim*16)))
def VAE_decoder(z):
	dim = 64
	kernel_size = 5
	print(z.shape)
	#x = Dense(dim*4*8*8, activation = 'relu')(z)
	#print(x.shape)
	x = Reshape((8, 8, 16))(z)
	#x = Conv2DTranspose(dim*4, (kernel_size, kernel_size), strides=(1, 1), activation='relu', padding='same', name='conv4')(x)
	#x = BatchNormalization()(x)
	#x = Conv2DTranspose(dim*2, (kernel_size, kernel_size), strides=(2, 2), activation='relu', padding='same', name='conv5')(x)
	#x = BatchNormalization()(x)
	#x = Conv2DTranspose(dim, (kernel_size, kernel_size), strides=(2, 2), activation='relu', padding='same', name='conv6')(x) 
	#x = BatchNormalization()(x)
	#x = Conv2DTranspose(dim, (kernel_size, kernel_size), strides=(2, 2), activation='relu', padding='same', name='conv7')(x) 
	
	x = Conv2D(dim*4, (kernel_size, kernel_size), activation='relu', padding='same', name='conv4')(x)
	x = UpSampling2D((2, 2), name='up1')(x)
	x = Conv2D(dim*2, (kernel_size, kernel_size), activation='relu', padding='same', name='conv5')(x)
	x = UpSampling2D((2, 2), name='up2')(x)
	x = Conv2D(dim, (kernel_size, kernel_size), activation='relu', padding='same', name='conv6')(x) 
	x = UpSampling2D((2, 2), name='up3')(x)

	#x = Conv2D(dim*16, (kernel_size, kernel_size), activation='relu', padding='same', name='conv7')(x) 
	#x = UpSampling2D((2, 2), name='up4')(x)
	decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same', name='recons')(x)
	print("decoded =", decoded.shape)
	return decoded

def VAE_encoder2(input_img):
	dim = 64
	kernel_size = 4
	x = Conv2D(dim, (kernel_size, kernel_size), strides=(2, 2), padding='same', name='conv1')(input_img) #nb_filter, nb_row, nb_col
	x = BatchNormalization()(x)
	x = LeakyReLU(alpha=0.2)(x)
	x = Conv2D(dim*2, (kernel_size, kernel_size), strides=(2, 2), padding='same', name='conv2')(x) #nb_filter, nb_row, nb_col
	x = BatchNormalization()(x)
	x = LeakyReLU(alpha=0.2)(x)
	x = Conv2D(dim*4, (kernel_size, kernel_size), strides=(2, 2), padding='same', name='conv3')(x) #nb_filter, nb_row, nb_col
	x = BatchNormalization()(x)
	x = LeakyReLU(alpha=0.2)(x)
	h = Flatten()(x)
	#h = Dense(512)(x)
	print(h.shape)
	#h = Flatten()(x)
	return h

def VAE_decoder2(z):
	dim = 64
	kernel_size = 4
	print(z.shape)
	x = Dense(dim*8*8*4, activation = 'relu')(z)
	#x = ReLU(x)

	#print(x.shape)
	x = Reshape((8, 8, dim*4))(x)
	#x = Conv2DTranspose(dim*4, (kernel_size, kernel_size), strides=(2, 2), activation='relu', padding='same', name='conv4')(x)
	#x = BatchNormalization(momentum=0.8)(x)
	#x = LeakyReLU(alpha=0.2)(x)
	x = Conv2DTranspose(dim*2, (kernel_size, kernel_size), strides=(2, 2), padding='same', name='conv5')(x)
	#x = BatchNormalization()(x)
	x = LeakyReLU(alpha=0.2)(x)
	x = Conv2DTranspose(dim, (kernel_size, kernel_size), strides=(2, 2), padding='same', name='conv6')(x) 
	#x = BatchNormalization()(x)
	x = LeakyReLU(alpha=0.2)(x)
	decoded = Conv2DTranspose(3, (kernel_size, kernel_size), strides=(2, 2), activation='sigmoid', padding='same', name='conv7')(x)
	
	#x = Conv2D(dim*4, (kernel_size, kernel_size), activation='relu', padding='same', name='conv4')(x)
	#x = UpSampling2D((2, 2), name='up1')(x)
	#x = Conv2D(dim*2, (kernel_size, kernel_size), activation='relu', padding='same', name='conv5')(x)
	#x = UpSampling2D((2, 2), name='up2')(x)
	#x = Conv2D(dim, (kernel_size, kernel_size), activation='relu', padding='same', name='conv6')(x) 
	#x = UpSampling2D((2, 2), name='up3')(x)
	#x = Conv2D(dim*16, (kernel_size, kernel_size), activation='relu', padding='same', name='conv7')(x) 
	#x = UpSampling2D((2, 2), name='up4')(x)
	#decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same', name='recons')(x)
	print("decoded =", decoded.shape)
	return decoded

# https://github.com/keras-team/keras/blob/master/examples/variational_autoencoder.py
def VAE():
	latent_dim = 1024
	epsilon_std = 1.0

	input_img = Input(shape=(64, 64, 3))
	h = VAE_encoder(input_img)

	z_mean = Dense(latent_dim, name='mean')(h)
	z_log_var = Dense(latent_dim, name='log_var')(h)
	print("encoded =", z_mean.shape)
	def sampling(args):
	    z_mean, z_log_var = args
	    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0., stddev=epsilon_std, seed=14)
	    print("epsilon =", epsilon.shape)
	    #std = logvar.mul(0.5).exp_()
		#eps = torch.cuda.FloatTensor(std.size()).normal_()
		#return eps.mul(std).add_(mu)

	    return z_mean + K.exp(z_log_var * 0.5) * epsilon

	z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
	decoded = VAE_decoder(z)

	KL_output = concatenate([z_mean, z_log_var], name='KLD')
	# instantiate VAE model
	model = Model(input_img, [decoded, KL_output])

	def loss_recons(real, pred):
		print("weighted_loss =", real.shape, pred.shape)
		loss_recons = losses.mean_squared_error(K.flatten(input_img), K.flatten(decoded))
		#loss_recons = K.mean(K.mean(losses.mean_squared_error(real, pred), axis = -1), axis = -1)
		return loss_recons

	def loss_KL(real, pred):
		z_mean, z_log_var = pred[:, :latent_dim], pred[:, latent_dim:]
		loss_KL = - 0.5 * K.sum(1.0 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
		#loss_KL = lamb * loss_KL
		#mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
		return loss_KL
	lamb = 3e-4
	model.compile(optimizer='adam', loss=[loss_recons, loss_KL], loss_weights=[1., lamb])#, metrics=["mse"])


	'''loss_recons = mse(K.flatten(input_img), K.flatten(decoded))
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(loss_recons + lamb * kl_loss)'''


	return model

def autoencoder_1():
	input_img = Input(shape=(64, 64, 3)) 

	x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img) #nb_filter, nb_row, nb_col
	x = MaxPooling2D((2, 2))(x)
	x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
	x = MaxPooling2D((2, 2))(x)
	x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
	encoded = MaxPooling2D((2, 2), padding='same')(x)
	print("encoded =", encoded.shape)

	x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
	x = UpSampling2D((2, 2))(x)
	x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
	x = UpSampling2D((2, 2))(x)
	print("x =", x.shape)

	x = Conv2D(16, (3, 3), activation='relu', padding='same')(x) 
	x = UpSampling2D((2, 2))(x)
	decoded = Conv2D(3, (5, 5), activation='sigmoid', padding='same')(x)

	print("decoded =", decoded.shape)
	print("shape of decoded", K.int_shape(decoded))

	autoencoder = Model(input_img, decoded)
	autoencoder.compile(optimizer='adadelta', loss='mean_squared_error') # 'binary_crossentropy'
	#autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
	return autoencoder

