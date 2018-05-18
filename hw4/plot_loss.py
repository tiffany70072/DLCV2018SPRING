import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys


def plot_VAE():
	filename = ['VAE2_recons_loss.npy', 'VAE2_KLD_loss.npy']
	title = ['MSE', "KLD"]
	x = [i for i in range(0, 200, 1)]
	f, (ax1, ax2) = plt.subplots(1, 2, figsize = (10, 5))
	ax = [ax1, ax2]
	for i in range(2):
		loss = np.load(filename[i])
		ax[i].plot(x, loss)
		ax[i].set_title(title[i])
		plt.ylabel('loss')
		plt.xlabel('epochs')
	plt.savefig(sys.argv[2] + 'fig1_2.jpg')

def plot_GAN():
	filename = "report/GAN_history.npy"
	history1 = np.load(filename)
	filename = "report/GAN_history2.npy"
	history2 = np.load(filename)[:1000]
	history = np.concatenate([history1, history2], axis = 0)
	smooth_n = 20
	x = [i for i in range(0, 20000 - smooth_n * 10, 10)]
	print(history1.shape, history2.shape)
	lossd_smooth = []
	lossd = history[:, 0]
	print(len(lossd))
	
	for i in range(len(lossd) - smooth_n):
		lossd_smooth.append(np.mean(np.array(lossd[i:i+smooth_n])))

	lossg_smooth = []
	lossg = history[:, 2]
	print(len(lossg))
	for i in range(len(lossg) - smooth_n):
		lossg_smooth.append(np.mean(np.array(lossg[i:i+smooth_n])))

	plt.plot(x, lossd_smooth)
	#plt.plot(x, history[:, 2])
	plt.plot(x, lossg_smooth)
	#plt.plot(x, history[:, 1]) # 'd_acc', 
	plt.title("GAN")
	plt.ylabel('loss')
	plt.xlabel('iteraions')
	plt.axis([0, 20000 - smooth_n, 0, 5])
	plt.legend(['d_loss', 'g_loss'], loc='upper left')

	#plt.axis([0, 20000 - smooth_n, 0, 100])
	#plt.legend('d_acc', loc='upper left')
	#plt.show()
	plt.savefig(sys.argv[2] + 'fig2_2.jpg')
	#history.append([self.d_loss[0], 100*self.d_loss[1], self.g_loss])

def plot_ACGAN():
	filename = "report/process_ACGAN"
	fin = open(filename, 'r')
	lossd = []
	lossg = []
	count = 0
	for line in fin:
		if count > 0 and count < 2010:
			d = line.split(' ')[13][:-2]
			g = line.split(' ')[18][:-2]
			lossd.append(float(d))
			lossg.append(float(g))
		count += 1

	smooth_n = 10
	x = [i for i in range(0, 20000 - smooth_n * 10, 10)]
	lossd_smooth = []
	lossd = np.array(lossd[:2000])
	lossd.astype(np.float32)
	
	for i in range(len(lossd) - smooth_n):
		lossd_smooth.append(np.mean(lossd[i:i+smooth_n]))

	lossg_smooth = []
	lossg = np.array(lossg[:2000])
	lossg.astype(np.float32)
	print(len(lossg))
	for i in range(len(lossg) - smooth_n):
		lossg_smooth.append(np.mean(lossg[i:i+smooth_n]))

	plt.plot(x, lossd_smooth)
	#plt.plot(x, history[:, 1]) # 'd_acc', 
	#plt.plot(x, history[:, 2])
	plt.plot(x, lossg_smooth)
	plt.title("ACGAN")
	plt.ylabel('loss')
	plt.xlabel('iterations')
	plt.axis([0, 20000 - smooth_n, 0, 3])
	plt.legend(['d_loss', 'g_loss'], loc='upper left')
	#plt.show()
	plt.savefig(sys.argv[2] + 'fig3_2.jpg')
	

if sys.argv[1] == 'VAE': plot_VAE()
if sys.argv[1] == 'GAN': plot_GAN()
if sys.argv[1] == 'ACGAN': plot_ACGAN()