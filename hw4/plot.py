import numpy as np
import scipy.misc

def output_32(pred, path, name):
	print('output_32')
	print("pred =", pred.shape)
	all_images = np.empty([4*64, 8*64, 3], dtype = np.float32)
	size = 64
	for i in range(4):
		for j in range(8):
			all_images[i*64:(i+1)*64, j*64:(j+1)*64] = pred[i*8+j]
	#scipy.misc.imsave(str(path) + str(name), all_images)
	scipy.misc.imsave(str(path) , all_images)
	print(str(path))
	#scipy.misc.imsave('output_test/' + str(name) + '.jpg' , all_images)

def output_20(pred, path, epoch):
	print('output_20')
	print("pred =", pred.shape)
	all_images = np.empty([2*64, 10*64, 3], dtype = np.float32)
	size = 64
	for i in range(2):
		for j in range(10):
			all_images[i*64:(i+1)*64, j*64:(j+1)*64] = pred[i*10+j]
	#scipy.misc.imsave(str(path) + str(epoch) + '.png', all_images)
	scipy.misc.imsave(str(path), all_images)
	print('output_test/' + str(epoch) + '.jpg')
	#scipy.misc.imsave('output_test/' + str(epoch) + '.jpg' , all_images)