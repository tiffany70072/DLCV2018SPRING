import numpy as np
import os
import scipy.misc
from skimage import io
from keras.utils import to_categorical

def import_image_branch1(folder, mode = 'load'):
	print("Import image...", folder)
	if mode == 'save':
		#assert folder == 'train' or folder == 'validation', 'invalid folder name'
		path = "hw3-train-validation/" + folder + "/"

		if folder == 'train': num = 2313
		elif folder == 'validation': num = 257
		sat  = np.empty([num, 512, 512, 3], dtype = np.float32)
		mask = np.zeros([num, 512, 512, 7], dtype = np.uint8)
		
		count = 0
		for filename in os.listdir(path):
			count += 1			
			img = (io.imread(path+filename))
			idx = int(filename[:4])
			img = img/255.0
			
			if filename[-3:] == 'jpg': sat[idx] = img
			elif filename[-3:] == 'png':
				img = 4*img[:, :, 0] + 2*img[:, :, 1] + img[:, :, 2]

				mask[idx, img == 0] = [1, 0, 0, 0, 0, 0, 0]  
				mask[idx, img == 1] = [0, 1, 0, 0, 0, 0, 0]  
				mask[idx, img == 2] = [0, 0, 1, 0, 0, 0, 0]  
				mask[idx, img == 3] = [0, 0, 0, 1, 0, 0, 0]  
				mask[idx, img == 4] = [0, 0, 0, 0, 0, 0, 1]  
				mask[idx, img == 5] = [0, 0, 0, 0, 1, 0, 0]  
				mask[idx, img == 6] = [0, 0, 0, 0, 0, 1, 0]  
				mask[idx, img == 7] = [0, 0, 0, 0, 0, 0, 1]  

			if count % 100 == 0: print(count)
		print("mask =", mask.shape)
		print(count)

	elif mode == 'load':
		sat = np.load('data/' + folder + '_sat.npy')
		mask = np.load('data/' + folder + '_mask.npy')

	print("sat =", sat.shape)
	print("mask =", mask.shape)
	return sat, mask

def import_image(folder):
	print("Import image...", folder)
	path = "hw3-train-validation/" + folder + "/"

	if folder == 'train': num = 2313
	elif folder == 'validation': num = 257
	sat  = np.empty([num, 512, 512, 3], dtype = np.float32)
	mask = np.zeros([num, 512, 512, 7], dtype = np.uint8)
		
	count = 0
	for filename in os.listdir(path):
		count += 1			
		img = (io.imread(path+filename))
		idx = int(filename[:4])
		img = img/255.0
		if count % 100 == 0: print(count)
		if filename[-3:] == 'jpg': sat[idx] = img

	file_list = [file for file in os.listdir(path) if file.endswith('.png')]
	file_list.sort()
	n_masks = len(file_list)
	assert n_masks == num
	masks = np.zeros([num, 512, 512, 7], dtype = np.uint8)
	for i, file in enumerate(file_list):
		mask = scipy.misc.imread(os.path.join(path, file))
		mask = (mask >= 128).astype(int)
		mask = 4 * mask[:, :, 0] + 2 * mask[:, :, 1] + mask[:, :, 2]
		masks[i, mask == 3] = [1, 0, 0, 0, 0, 0, 0]  # (Cyan: 011) Urban land 
		masks[i, mask == 6] = [0, 1, 0, 0, 0, 0, 0]  # (Yellow: 110) Agriculture land 
		masks[i, mask == 5] = [0, 0, 1, 0, 0, 0, 0]  # (Purple: 101) Rangeland 
		masks[i, mask == 2] = [0, 0, 0, 1, 0, 0, 0]  # (Green: 010) Forest land 
		masks[i, mask == 1] = [0, 0, 0, 0, 1, 0, 0]  # (Blue: 001) Water 
		masks[i, mask == 7] = [0, 0, 0, 0, 0, 1, 0]  # (White: 111) Barren land 
		masks[i, mask == 0] = [0, 0, 0, 0, 0, 0, 1]  # (Black: 000) Unknown 
		masks[i, mask == 4] = [0, 0, 0, 0, 0, 0, 1]  # (Black: 000) Unknown 
		if i % 100 == 0: print(i)
	print("sat =", sat.shape)
	print("mask =", masks.shape)
	return sat, masks

def import_test_image(path, mode):
	print("Import testing image...", path)
	if mode == 'save':
		#assert folder == 'train' or folder == 'validation', 'invalid folder name'
		sat = []
		idx = []
		count = 0
		file_list = [file for file in os.listdir(path) if file.endswith('.jpg')]
		print(file_list)
		for filename in file_list:
			count += 1			
			img = list(io.imread(path+filename))
			img = img/255.0
			sat.append(img)
			idx.append(filename[:4])
			print(filename[:4])
			
		print(count)
	sat = np.array(sat)

	print("sat =", sat.shape)
	print("idx =", len(idx))
	return sat, idx

def main():
	import_image('train')

if __name__ == '__main__':
	main()

