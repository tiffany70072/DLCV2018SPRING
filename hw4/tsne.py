import numpy as np
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys

h = np.load("report/VAE_problem5_h.npy")#[:100]
h = h[:500]
x_embedded = TSNE(n_components=2).fit_transform(h)
print("embedded =", x_embedded.shape)

gender = []
import csv
with open(sys.argv[1] + 'test.csv', 'rt') as csvfile:
	spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
	flag = 0
	for row in spamreader:
		if flag == 1:
			new_row = row[0].split(',')
			gender.append(new_row[8])
		else: flag = 1
print(len(gender), gender[:10])
#print np.sum(np.array(gender))

for i in range(500):
	if float(gender[i]) == 0:
		color = 'r'
	else: color = 'b'
	plt.scatter(x_embedded[i][0], x_embedded[i][1], c = color)

plt.savefig(sys.argv[2] + 'fig1_5.jpg')