import numpy as np
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys

def plot(mode):
	if mode == 'cnn': latent = np.load("../report/p1_features.npy")
	elif mode == 'rnn': latent = np.load("../report/p2_features.npy")
	
	np.random.seed(0)
	embedded = TSNE(n_components=2, random_state=14, init='pca').fit_transform(latent)
	print("embedded =", embedded.shape)
	y = np.load('data/y_valid.npy')
	print('y =', y.shape)
	y = np.argmax(y, axis = -1)
	print('y =', y.shape)
	for i in range(517):
		if y[i] > 0:
			plt.scatter(embedded[i][0], embedded[i][1], c = 'C' + str(y[i]-1))
		else:
			plt.scatter(embedded[i][0], embedded[i][1], c = 'k')

	plt.savefig("../report/" + mode + "-based features 2")
	plt.show()

plot('cnn')
plot('rnn')