import numpy as np
import os
import scipy.misc
from skimage import io
import csv
#from keras.utils import to_categorical

def import_image(path = None, folder = None):
	# train: 40000, test: 2621
	if path == None: path = "hw4_data/" + folder + "/" 
	else: path = path + "test/"
	file_list = [file for file in os.listdir(path) if file.endswith('.png')]
	file_list.sort()
	num = len(file_list)
	print("num =", num)
	
	imgs = np.zeros([num, 64, 64, 3], dtype = np.float32)
	for i, file in enumerate(file_list):
		imgs[i] = scipy.misc.imread(os.path.join(path, file))
		if i % 500 == 0: print(i)
	imgs = imgs/255.0
	return imgs
		
def import_image_features(folder):
	# train: 40000, test: 2621
	# [ 7393.  7323. 11197.  5748. 10905. 15452. 15801. 14878. 16864. 16442.
	# 10111. 12948. 19605.]
	# Bangs,Big_Lips,Black_Hair,Blond_Hair,Brown_Hair,Heavy_Makeup,High_Cheekbones,Male,
	# Mouth_Slightly_Open,Smiling,Straight_Hair,Wavy_Hair,Wearing_Lipstick
	if folder == "train": num = 40000
	if folder == "test": num = 2621
	features = np.zeros([num, 1], np.uint8)
	ID = 10 # Smiling
	count = 0
	with open('hw4_data/' + folder + '.csv', 'r') as csvfile:
		spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
		flag = 0
		for row in spamreader:
			if flag == 1:
				new_row = row[0].split(',')
				features[count][0] = int(float(new_row[ID]))
				#print(features[count][0])
				count += 1
			else: flag = 1

	print(count)
	return features
		
def main():
	import_image('test')

if __name__ == '__main__':
	main()

