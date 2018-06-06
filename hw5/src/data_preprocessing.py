import numpy as np
from import_data import import_trimmed
from import_data import import_test_trimmed, import_test_groudtruth
import sys

#from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing import image
from keras.models import Model, Sequential
from keras.layers import Dense, GlobalAveragePooling2D, concatenate, Input, Lambda, Dropout, Concatenate, Flatten
from keras import backend as K
from keras.utils.np_utils import to_categorical
from keras import optimizers

from keras.preprocessing.image import ImageDataGenerator

from tensorflow import ConfigProto
import tensorflow as tf
#config = ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.2
#sess = tf.Session(config=config)
import keras
#keras.backend.set_session(sess)

class DataPreprocessing(object):
	def __init__(self):
		self.features_length = 2048
		#self.load_xy()
		#self.load_inceptionV3()
		#model.load_resnet50()
		#self.load_features()

	def load_xy(self):
		'''self.x_train, labels = import_four_trimmed('train') # videos, labels
		self.y_train = to_categorical(labels, num_classes = 11)
		self.x_valid, labels = import_four_trimmed('valid') # videos, labels
		self.y_valid = to_categorical(labels, num_classes = 11)

		print('y_train =', self.y_train[:10])
		print('y_valid =', self.y_valid[:10])'''

		self.y_train = np.load('data/y_train.npy')
		self.y_valid = np.load('data/y_valid.npy')
		self.idx_train = np.load('data/idx_train.npy')
		self.idx_valid = np.load('data/idx_valid.npy')
		self.x_train = np.load('data/x_train.npy')
		self.x_valid = np.load('data/x_valid.npy')
		print("dtype =", self.x_train.dtype, self.y_train.dtype, self.idx_train.dtype)

	def output_accuracy(self, cnn_model):
		pred = cnn_model.predict(self.features_valid)
		real = self.y_valid
		print('pred =', pred.shape)

		correct = np.sum(np.argmax(real, axis = 1) == np.argmax(pred, axis = 1))
		print("classification accuracy =", correct, correct/float(real.shape[0]))
		print("real, pred", np.argmax(real[:10], axis = 1), np.argmax(pred[:10], axis = 1))
		return correct/float(real.shape[0])
	
	def load_inceptionV3(self):
		# https://github.com/harvitronix/five-video-classification-methods
		'''print('\nLoading model')
		base_model = InceptionV3(weights='imagenet', include_top=False)
		print('Feature selection')
		self.features = base_model.predict(self.x_train.reshape([-1, 240, 320, 3]))
		self.features_valid = base_model.predict(self.x_valid.reshape([-1, 240, 320, 3]))
		print('Del')
		del base_model
		del self.x_train
		del self.x_valid
		
		print("features =", self.features.shape)
		self.features = self.features.reshape([-1, 4, 6, 8, 2048])
		self.features_valid = self.features_valid.reshape([-1, 4, 6, 8, 2048])
		np.save('features_train', self.features)
		np.save('features_valid', self.features_valid)
		#np.save('y_train', self.y_train)'''
		self.features = np.load('features_train.npy')
		self.features_valid = np.load('features_valid.npy')
		print(self.features[0][1][0][3][:10])
		print(self.features[1][1][0][3][:10])
		self.features = self.features #[:, 1]
		self.features_valid = self.features_valid #[:, 1]

		#nn_model = Sequential()
		#cnn_model.add(Dense(2048, activation='relu'))
		#cnn_model.add(Dropout(0.5))
		#cnn_model.add(Dense(self.nb_classes, activation='softmax'))
		print('Build top layers')
		'''inputs = Input(shape = (4, 6, 8, 2048)) # feature inputs
		cnn_model = Sequential()
		#cnn_model.add(Lambda(self.concat_layer, input_shape=(4, 6, 8, 2048)))
		#cnn_model.add(ConcatLayer, input_shape=(4, 6, 8, 2048))
		cnn_model.add(Dense(11, activation='softmax'))
		cnn_model.summary()'''
		
	def load_resnet50(self):
		print('\nLoading model')
		base_model = ResNet50(weights='imagenet', include_top=False)
		print('Feature selection')
		self.features = base_model.predict(self.x_train, verbose=1)
		self.features_valid = base_model.predict(self.x_valid, verbose=1)

		print('Del')
		del base_model
		del self.x_train
		del self.x_valid

		pool_layer = Sequential()
		pool_layer.add(GlobalAveragePooling2D(input_shape=(1, 1, 2048)))
		pool_layer.summary()

		self.features = pool_layer.predict(self.features) 
		self.features_valid = pool_layer.predict(self.features_valid) 
		
		print("features =", self.features.shape, self.features_valid.shape)
		np.save('data/features_train', self.features)
		np.save('data/features_valid', self.features_valid)

	def get_seq(self, features, seq_len, idx):
		seq = np.zeros([seq_len, 2048])
		count = 0
		for i in range(idx.shape[0]):
			seq[i] = np.mean(features[count:count+idx[i]], axis = 0)
			if i == 100: print(np.mean(features[count:count+idx[i]], axis = 0).shape)
			count += idx[i]
		assert count == features.shape[0]
		return seq

	def load_features(self):
		print("Loading features")
		self.features = np.load('data/features_train.npy')
		self.features_valid = np.load('data/features_valid.npy')
		self.y_train = np.load('data/y_train.npy')
		self.y_valid = np.load('data/y_valid.npy')
		self.idx_train = np.load('data/idx_train.npy')
		self.idx_valid = np.load('data/idx_valid.npy')
		print("features =", self.features.shape,  self.features_valid.shape)
		print("y =",        self.y_train.shape,   self.y_valid.shape)
		print("idx =",      self.idx_train.shape, self.idx_valid.shape)

		self.features = self.get_seq(self.features, 3236, self.idx_train)
		self.features_valid = self.get_seq(self.features_valid, 517, self.idx_valid)
		print("features sequence =", self.features.shape,  self.features_valid.shape)

	def build_cnn_model(self):
		inputs = Input(shape = (2048, ))
		h = Dense(1024, activation = 'relu', name='FC1')(inputs) #(concat)
		#h = Dropout(0.5)(h)
		#h = Dense(2048, activation = 'relu', name="FC2")(h) #(concat)
		h = Dropout(0.5)(h)
		prediction = Dense(11, activation='softmax', name="output")(h)
		cnn_model = Model(inputs=inputs, outputs=prediction)
		cnn_model.summary()

		print('\nModel fitting')
		opt = optimizers.Adam(lr = 1E-4)
		#opt = optimizers.SGD(lr = 1E-4)
		cnn_model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
		return cnn_model
		
	def train(self):
		cnn_model = self.build_cnn_model()
		history = cnn_model.fit(self.features, self.y_train, epochs = 40, validation_data = (self.features_valid, self.y_valid))
		
		print("history =", history.history.keys())
		h1 = history.history['loss']
		h2 = history.history['val_loss']
		h3 = history.history['acc']
		h4 = history.history['val_acc']
		np.save('../report/p1_loss', h1)
		np.save('../report/p1_val_loss', h2)
		np.save('../report/p1_acc', h3)
		np.save('../report/p1_val_acc', h4)
			
		print('Save model')
		cnn_model.save('saved_model/p1_weights_2.h5')

	def test(self):
		print("Testing")
		cnn_model = self.build_cnn_model()
		cnn_model.load_weights('saved_model/p1_cnn_weights.h5')
		pred = cnn_model.predict(self.features_valid)
		print("\nEval =", cnn_model.evaluate(self.features_valid, self.y_valid))
		print("pred.shape =", pred.shape)
		pred = np.argmax(pred, axis = -1)
		print("pred.shape =", pred.shape)

		fout = open(sys.argv[3] + 'p1_valid.txt', 'w')
		for i in range(pred.shape[0]):
			fout.write("%d\n" %pred[i])
		fout.close()

	def get_cnn_features(self):
		print("Get cnn features")
		inputs = Input(shape = (2048, ))
		h = Dense(2048, activation = 'relu')(inputs) #(concat)
		model = Model(inputs=inputs, outputs=h)
		model.load_weights('saved_model/p1_cnn_weights.h5', by_name = True)

		features_top = model.predict(self.features_valid)
		print("features_top =", features_top.shape)
		np.save("../report/p1_features", features_top)

	def load_test(self):
		x_valid, self.idx_valid = import_test_trimmed() 
		labels = import_test_groudtruth()
		self.y_valid = to_categorical(labels, num_classes = 11)

		print('\nLoading model')
		base_model = ResNet50(weights='imagenet', include_top=False)
		print('Feature selection')
		features = base_model.predict(x_valid, verbose=1)
		del base_model
		del x_valid
		
		pool_layer = Sequential()
		pool_layer.add(GlobalAveragePooling2D(input_shape=(1, 1, 2048)))
		pool_layer.summary()

		self.features_valid = pool_layer.predict(features) 
		self.features_valid = self.get_seq(self.features_valid, 517, self.idx_valid)
		
def main():
	model = DataPreprocessing()
	#model.build_cnn_model()
	#model.train()
	model.load_test()
	model.test()
	#model.get_cnn_features()

if __name__ == "__main__":
	main()

