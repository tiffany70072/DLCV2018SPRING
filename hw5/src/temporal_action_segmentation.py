import numpy as np
import os
import scipy.misc
import sys

from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing import image
from keras.models import Model, Sequential
from keras.layers import Dense, GlobalAveragePooling2D, concatenate, Input, Lambda, Dropout, Concatenate, Flatten
from keras.layers import LSTM
from keras import backend as K
#from keras.utils import to_categorical
from keras.utils.np_utils import to_categorical
from keras.engine.topology import Layer
from keras import optimizers
from keras.preprocessing.sequence import pad_sequences
from keras.layers.wrappers import TimeDistributed

from tensorflow import ConfigProto
import tensorflow as tf
config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
sess = tf.Session(config=config)
import keras
keras.backend.set_session(sess)

class TemporalActionSegmentation(object):
	def __init__(self):
		self.nb_classes = 11
		self.features_length = 2048
		self.train_num = 3236
		self.valid_num = 517
		self.max_length = 300  
		self.interval_length = 250
		self.is_train = False

	def cut(self, one_video):
		pos = 0
		count = 0
		num = int((len(one_video)-1-self.max_length+self.interval_length)/self.interval_length) + 1
		print("total =", len(one_video), ", num =", num)
		try:
			if one_video[0].ndim == 3: 
				cut_video = np.empty([num, self.max_length, 240, 320, 3], dtype = np.uint8)
		except AttributeError:
			cut_video = np.empty([num, self.max_length], dtype = np.uint8)
		while pos + self.max_length < len(one_video):
			cut_video[count] = np.array(one_video[pos:pos+self.max_length])
			print("pos =", pos, ", count =", count)
			pos += self.interval_length
			count += 1
			
		cut_video[count] = np.array(one_video[len(one_video)-self.max_length:])
		print("pos =", len(one_video)-self.max_length, ", count =", count)
		return cut_video


	def import_full(self, data):
		print('\ndata type =', data)
		path = '../HW5_data/FullLengthVideos/videos/' + data + '/'
		if data == 'valid': path = sys.argv[1]

		video_list = [file for file in os.listdir(path)]
		video_list.sort()
		print(video_list, len(video_list))
		if data == 'valid': 
			self.category = video_list
			print("category =", self.category)

		if data == 'valid': self.idx_len = np.empty([5], dtype = np.uint16)
		videos = np.empty([0, self.max_length, 240, 320, 3], dtype = np.uint8)
		for i, video in enumerate(video_list):
			image_list = [file for file in os.listdir(path + video + '/')]
			
			image_list.sort()
			print('len', i, len(image_list), video)
			#print(image_list[:3])
			assert len(image_list) > self.max_length

			if data == 'valid': 
				self.idx_len[i] = len(image_list)
			one_video = []
			for j, image in enumerate(image_list):
				one_image = scipy.misc.imread(path + '/' + video + '/' + image)
				one_video.append(one_image)
			one_video = self.cut(one_video)
			videos = np.concatenate([videos, one_video], axis = 0)

		print("videos =", videos.shape, videos[0][0][0][0])
		return videos

	def import_full_labels(self, data):
		print('\ndata type =', data)
		
		path = '../HW5_data/FullLengthVideos/labels/' + data + '/'

		video_list = [file for file in os.listdir(path)]
		video_list.sort()
		print(video_list, len(video_list))
		labels = np.empty([0, self.max_length], dtype = np.uint8)
		if data == 'valid': 
			self.idx_begin = np.empty([5], dtype = np.uint8)
			self.idx_end = np.empty([5], dtype = np.uint8)
			self.idx_begin[0] = 0
		
		for i, video in enumerate(video_list):
			#image_list = [file for file in os.listdir(path + video + '/')]
			#image_list.sort()
			label = []
			fin = open(path + video, 'r')
			j = 0
			for one_label in fin:
				j += 1
				label.append(int(one_label[0]))
			
			print("j =", j)
			label = self.cut(label)
			labels = np.concatenate([labels, label], axis = 0)

			if data == 'valid':
				if i != 0: 
					self.idx_begin[i] = self.idx_end[i-1] + 1
				self.idx_end[i] = self.idx_begin[i] + label.shape[0] - 1
			
		if data == 'valid': print('idx =', self.idx_begin, self.idx_end)
		print("labels =", labels.shape, labels[0][:10])

		return labels

	def load_xy(self):
		if self.is_train == True:
			self.y_train = self.import_full_labels('train')
			self.y_train = to_categorical(self.y_train, num_classes = 11)
			self.y_train = self.y_train.reshape([-1, self.max_length, 11])
			print('train =', self.features.shape, self.y_train.shape, self.y_train[0][:10])

			self.y_valid = self.import_full_labels('valid')
			self.y_valid = to_categorical(self.y_valid, num_classes = 11)
			self.y_valid = self.y_valid.reshape([-1, self.max_length, 11])
			
			print('valid =', self.features_valid.shape, self.y_valid.shape, self.y_valid[0][:10])
		
		#np.save('data/y_train', self.y_train)
		#np.save('data/y_valid', self.y_valid)
		#np.save('data/x_train', self.x_train)
		#np.save('data/x_valid', self.x_valid)
		#np.save('data/idx_train', self.idx_train)
		#np.save('data/idx_valid', self.idx_valid)

	def load_model(self):
		# https://github.com/harvitronix/five-video-classification-methods
		print('\nLoading model')
		base_model = ResNet50(weights='imagenet', include_top=False)

		if self.is_train == True:
			print('Feature selection')
			self.x_train = self.import_full('train') 
			self.features = base_model.predict(self.x_train.reshape([-1, 240, 320, 3]))
			del self.x_train
			print("features =", self.features.shape)

		print('Feature selection')
		self.x_valid = self.import_full('valid')
		self.features_valid = base_model.predict(self.x_valid.reshape([-1, 240, 320, 3]))
		print('Del')
		del base_model
		del self.x_valid

		pool_layer = Sequential()
		pool_layer.add(GlobalAveragePooling2D(input_shape=(1, 1, 2048)))
		pool_layer.summary()

		if self.is_train == True:
			self.features = pool_layer.predict(self.features) #[:, 1]
			print("features =", self.features.shape, self.features_valid.shape)
			self.features = self.features.reshape([-1, self.max_length, 2048])


		self.features_valid = pool_layer.predict(self.features_valid) #[:, 1]
		self.features_valid = self.features_valid.reshape([-1, self.max_length, 2048])
		#np.save('data/features3_train', self.features)
		#np.save('data/features3_valid', self.features_valid)
		#np.save('features_train', self.features)
		#np.save('features_valid', self.features_valid)

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
		for i in range((self.idx_train.shape[0])): count[int(int(self.idx_train[i])/40)] += 1
		print("count =", count)
		print("max_length =", np.max(self.idx_train), np.max(self.idx_valid))

		#self.features_seq = np.zeros([3236, max_length, 2048])
		#self.features_valid_seq = np.zeros([517, max_length, 2048])

		def get_seq(features, seq_len, idx, max_length):
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

		self.features = get_seq(self.features, 3236, self.idx_train, max_length)
		self.features_valid = get_seq(self.features_valid, 517, self.idx_valid, max_length)
		print("features sequence =", self.features.shape,  self.features_valid.shape)

	def build_model(self):
		print('Build top layers')
		self.input_shape = (self.max_length, self.features_length)
		
		model = Sequential()
		#model.add(LSTM(2048, return_sequences=True, input_shape=self.input_shape, dropout=0.5, name="LSTM1"))
		#model.add(LSTM(2048, return_sequences=True, dropout=0.5, name="LSTM2"))
		model.add(LSTM(2048, return_sequences=True, input_shape=self.input_shape, dropout=0.5, name="LSTM1"))
		#model.add(Dense(512*2, activation='relu', name="FC1"))
		model.add(TimeDistributed(Dense(512*2, activation='relu', name="FC2")))
		model.add(Dropout(0.5))
		#model.add(Dense(512*2, activation='relu'))
		#model.add(Dropout(0.5))
		model.add(TimeDistributed(Dense(self.nb_classes, activation='softmax', name="output")))

		model.summary()
		opt = optimizers.Adam(lr = 1E-4)
		model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
		return model
		
	def train(self):
		rnn_model = self.build_model()
		#rnn_model.load_weights('../saved_model/p2_weights.h5', by_name = True)
		print('\nModel fitting')
		history = rnn_model.fit(self.features, self.y_train, epochs = 40, validation_data = (self.features_valid, self.y_valid))
		
		# save history and model
		#print("history =", history.history)
		print("history =", history.history.keys())
		h1 = history.history['loss']
		h2 = history.history['val_loss']
		h3 = history.history['acc']
		h4 = history.history['val_acc']
		np.save('../report/p3_loss', h1)
		np.save('../report/p3_val_loss', h2)
		np.save('../report/p3_acc', h3)
		np.save('../report/p3_val_acc', h4)

		print('Save model')
		rnn_model.save('saved_model/p3_weights.h5')

	def reconstruct_seq(self, arr):
		seq = [[], [], [], [], []]
		for i in range(5):
			for j in range(self.idx_begin[i], self.idx_end[i]+1, 1):
				if j == self.idx_begin[i]:
					begin = 0
					end = self.max_length
				elif j == self.idx_end[i]:
					begin = self.max_length - (self.idx_len[i] - (self.max_length + (j-self.idx_begin[i]-1)*self.interval_length))
					end = self.max_length
				else: 
					begin = self.max_length - self.interval_length
					end = self.max_length
				print("begin, end =", begin, end)
				for k in range(begin, end):
					seq[i].append(arr[j][k])
			print("i =", i, ", len =", len(seq[i]), self.idx_len[i])
		return seq

	def test(self):
		print("Testing")
		rnn_model = self.build_model()
		rnn_model.load_weights('saved_model/p3_weights.h5')
		pred = rnn_model.predict(self.features_valid)
		print("pred.shape =", pred.shape)
		pred = np.argmax(pred, axis = -1)
		real = np.argmax(self.y_valid, axis = -1)
		print("pred.shape =", pred.shape)

		pred = self.reconstruct_seq(pred)
		real = self.reconstruct_seq(real)

		for i in range(5):
			fout = open(sys.argv[2] + self.category[i] + '.txt', 'w')
			for j in range(len(pred[i])):
				#print(pred[i])
				fout.write("%d\t" %real[i][j])
				fout.write("%d\n" %pred[i][j])
			fout.close()

def main():
	model = TemporalActionSegmentation()
	#model.build_model()
	
	model.load_model()
	model.load_xy()
	
	#model.load_features() # load the features, output after the base model
	if model.is_train == True:
		model.train()

	model.test()
	#model.get_rnn_features()

if __name__ == "__main__":
	main()

