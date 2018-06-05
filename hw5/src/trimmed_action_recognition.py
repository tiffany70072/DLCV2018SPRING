import numpy as np
from import_data import import_trimmed
from import_data import import_test_trimmed, import_test_groudtruth
import sys

#from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing import image
from keras.models import Model, Sequential
from keras.layers import Dense, GlobalAveragePooling2D, concatenate, Input, Lambda, Dropout, Concatenate, Flatten
from keras.layers import LSTM
from keras import backend as K
from keras.utils.np_utils import to_categorical
from keras import optimizers

from tensorflow import ConfigProto
import tensorflow as tf
config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
sess = tf.Session(config=config)
import keras
keras.backend.set_session(sess)

class TrimmedActionRecognition(object):
	def __init__(self):
		self.nb_classes = 11
		self.features_length = 2048
		self.train_num = 3236
		self.valid_num = 517
		self.max_length = 80  

	def load_xy(self):
		self.x_train, labels, self.idx_train = import_trimmed('train') # videos, labels
		print('y_train_0 =', labels[:10])
		self.y_train = to_categorical(labels, num_classes = 11)

		self.x_valid, labels, self.idx_valid = import_trimmed('valid') # videos, labels
		self.y_valid = to_categorical(labels, num_classes = 11)

		print('y_train =', self.y_train[:10])
		print('y_valid =', self.y_valid[:10])
		
		#np.save('data/y_train', self.y_train)
		#np.save('data/y_valid', self.y_valid)
		#np.save('data/x_train', self.x_train)
		#np.save('data/x_valid', self.x_valid)
		#np.save('data/idx_train', self.idx_train)
		#np.save('data/idx_valid', self.idx_valid)

	def load_model(self):
		# https://github.com/harvitronix/five-video-classification-methods
		print('\nLoading model')
		#base_model = InceptionV3(weights='imagenet', include_top=False)
		print('Feature selection')
		self.features = base_model.predict(self.x_train.reshape([-1, 240, 320, 3]))
		self.features_valid = base_model.predict(self.x_valid.reshape([-1, 240, 320, 3]))
		print('Del')
		del base_model
		del self.x_train
		del self.x_valid
		
		print("features =", self.features.shape)
		#print(self.features[0][1][0][3][:10])
		#print(self.features[1][1][0][3][:10])

		pool_layer = Sequential()
		pool_layer.add(GlobalAveragePooling2D(input_shape=(6, 8, 2048)))
		pool_layer.summary()

		self.features = pool_layer.predict(self.features) #[:, 1]
		self.features_valid = pool_layer.predict(self.features_valid) #[:, 1]

		print("features =", self.features.shape, self.features_valid.shape)
		#np.save('data/features3_train', self.features)
		#np.save('data/features3_valid', self.features_valid)
		#np.save('features_train', self.features)
		#np.save('features_valid', self.features_valid)

	def get_seq(self, features, seq_len, idx, max_length):
		seq = np.zeros([seq_len, max_length, 2048])
		count = 0
		for i in range(idx.shape[0]):
			if idx[i] > max_length:
				m = int(idx[i]/max_length)+1
				base = max_length - int(idx[i]/m)
				print("idx =", idx[i], ", idx/m =", int(idx[i]/m), ", base =", base)
				for j in range(int(idx[i]/m)):
					seq[i][base+j] = features[count+j*m]
				count += idx[i]
			else:
				base = max_length - idx[i]
				for j in range(idx[i]):
					seq[i][base+j] = features[count+j]
				count += idx[i]
		assert count == features.shape[0]
		print("valid total length =", count, features.shape[0])
		return seq

	def load_features(self):
		self.features = np.load('data/features_train.npy')
		self.features_valid = np.load('data/features_valid.npy')
		self.y_train = np.load('data/y_train.npy')
		self.y_valid = np.load('data/y_valid.npy')
		self.idx_train = np.load('data/idx_train.npy')
		self.idx_valid = np.load('data/idx_valid.npy')
		print("features =", self.features.shape,  self.features_valid.shape)
		print("y =",        self.y_train.shape,   self.y_valid.shape)
		print("idx =",      self.idx_train.shape, self.idx_valid.shape)

		max_length = 80 #np.max(self.idx_train)
		count = [0, 0, 0, 0, 0, 0]
		for i in range((self.idx_train.shape[0])):
			count[int(int(self.idx_train[i])/40)] += 1
		print("count =", count)
		print("max_length =", np.max(self.idx_train), np.max(self.idx_valid))
	
		#self.features_seq = np.zeros([3236, max_length, 2048])
		#self.features_valid_seq = np.zeros([517, max_length, 2048])

		self.features = self.get_seq(self.features, 3236, self.idx_train, max_length)
		self.features_valid = self.get_seq(self.features_valid, 517, self.idx_valid, max_length)
		print("features sequence =", self.features.shape,  self.features_valid.shape)

	def build_model(self):
		print('Build top layers')
		self.input_shape = (self.max_length, self.features_length)
		
		model = Sequential()
		#model.add(LSTM(2048, return_sequences=True, input_shape=self.input_shape, dropout=0.5, name="LSTM1"))
		#model.add(LSTM(2048, return_sequences=False, dropout=0.5, name="LSTM2"))
		model.add(LSTM(2048, return_sequences=False, input_shape=self.input_shape, dropout=0.5, name="LSTM1"))
		model.add(Dense(512*2, activation='relu', name="FC1"))
		model.add(Dropout(0.5))
		#model.add(Dense(512*2, activation='relu'))
		#model.add(Dropout(0.5))
		model.add(Dense(self.nb_classes, activation='softmax', name="output"))

		model.summary()
		#opt = optimizers.SGD(lr = 1E-4)
		opt = optimizers.Adam(lr = 1E-4)
		model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
		return model
		
	def train(self):
		rnn_model = self.build_model()
		print('\nModel fitting')
		history = rnn_model.fit(self.features, self.y_train, epochs = 40, validation_data = (self.features_valid, self.y_valid))
		
		# save history and model
		print("history =", history.history.keys())
		h1 = history.history['loss']
		h2 = history.history['val_loss']
		h3 = history.history['acc']
		h4 = history.history['val_acc']
		np.save('../report/p2_loss', h1)
		np.save('../report/p2_val_loss', h2)
		np.save('../report/p2_acc', h3)
		np.save('../report/p2_val_acc', h4)

		print('Save model')
		rnn_model.save('saved_model/p2_rnn_weights.h5')

	def test(self):
		print("Testing")
		rnn_model = self.build_model()
		rnn_model.load_weights('saved_model/p2_rnn_weights.h5')
		print("\nEval =", rnn_model.evaluate(self.features_valid, self.y_valid))
		pred = rnn_model.predict(self.features_valid)
		print("pred.shape =", pred.shape)
		pred = np.argmax(pred, axis = -1)
		print("pred.shape =", pred.shape)

		fout = open(sys.argv[3] + '/p2_result.txt', 'w')
		for i in range(pred.shape[0]):
			fout.write("%d\n" %pred[i])
		fout.close()

	def get_rnn_features(self):
		print("Get rnn features")
		model = Sequential()
		#model.add(LSTM(2048, return_sequences=True, input_shape=self.input_shape, dropout=0.5, name="LSTM1"))
		#model.add(LSTM(2048, return_sequences=False, dropout=0.5, name="LSTM2"))
		model.add(LSTM(2048, return_sequences=False, input_shape=self.input_shape, dropout=0.5, name="LSTM1"))
		model.add(Dense(512*2, activation='relu', name="FC1"))
		#model.add(Dropout(0.5))

		model.load_weights('saved_model/p2_rnn_weights.h5', by_name = True)
		features_top = model.predict(self.features_valid)
		print("features_top =", features_top.shape)
		np.save("../report/p2_features", features_top)

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
		self.features_valid = self.get_seq(self.features_valid, self.y_valid.shape[0], self.idx_valid, 80)

def main():
	model = TrimmedActionRecognition()
	
	#model.load_xy()
	#model.load_model()
	
	#model.load_features() # load the features, output after the base model
	#model.build_model()
	#model.train()

	model.load_test()
	model.test()
	#model.get_rnn_features()

if __name__ == "__main__":
	main()

