import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys

def plot(p, epochs):
	print('plot', p)
	path = '../report/'
	filename = [p+'_loss.npy', p+'_val_loss.npy', p+'_acc.npy', p+'_val_acc.npy']
	title = ["Loss", "Accuracy"]
	x = [i for i in range(epochs)]
	f, (ax1, ax2) = plt.subplots(1, 2, figsize = (10, 5))
	ax = [ax1, ax2]
	for i in range(2):
		h1 = np.load(path + filename[i*2])
		h2 = np.load(path + filename[i*2+1])
		ax[i].plot(x, h1)
		ax[i].plot(x, h2)
		ax[i].set_title(title[i])
		#plt.ylabel()
		#ax[i].xlabel('epochs')

	ax1.set_xlim([0, epochs])
	ax1.set_ylim([0, 2])
	ax2.set_xlim([0, epochs])
	ax2.set_ylim([0, 1])
	plt.xlabel('epochs')
	plt.legend(['train', 'valid'], loc='upper left')
	plt.savefig('../report/' + p + '_learning_curve_3')
	plt.show()

#plot('p1', 40)
plot('p3', 40)