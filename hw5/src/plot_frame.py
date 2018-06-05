import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os

def compute_accuracy():
	path = "../report/"
	answers = [file for file in os.listdir(path) if file.endswith('.txt')]
	answers.sort()
	answers = answers[:5]
	print('file =', answers)
	
	for i, answer in enumerate(answers):
		fin = open(path + answer, 'r')
		pred = []
		real = []
		for line in fin:
			real.append(int(line[0]))
			pred.append(int(line[2]))
		real = np.array(real)
		pred = np.array(pred)


		correct = np.sum(real == pred)
		print("accuracy =", correct, correct/float(real.shape[0]))
		
'''
accuracy = 1419 0.66308411215
accuracy = 604 0.643923240938
accuracy = 417 0.486581096849
accuracy = 416 0.514215080346
accuracy = 817 0.600735294118
'''

def plot_one_line(arr, y, begin, end):
	for i in range(begin, end):
		if arr[i] > 0: plt.scatter(i-begin, y, c = 'C' + str(arr[i]-1))
		else: plt.scatter(i-begin, y, c = 'k')

def plot():
	path = "../report/"
	answers = [file for file in os.listdir(path) if file.endswith('.txt')]
	answers.sort()
	answers = answers[0]
	print('file =', answers)
	
	fin = open(path + answers, 'r')
	pred = []
	real = []
	for line in fin:
		real.append(int(line[0]))
		pred.append(int(line[2]))
	
	end = 300
	begin = 00
	plt.figure(figsize=(20, 3))
	plt.title = ["Loss", "Accuracy"]
	#x = [i for i in range(n)]
	
	plot_one_line(arr = pred, y = 0, begin = begin, end = end)
	plot_one_line(arr = real, y = 1, begin = begin, end = end)
	plt.y_lim = (-0.5, 1.5)
	
	plt.show()
	plt.savefig('../report/plot_frame_0_3')

compute_accuracy()
#plot()