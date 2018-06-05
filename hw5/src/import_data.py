import sys
sys.path.insert(0, '../')
import reader
import numpy as np
import scipy.misc

def import_four_trimmed(data = 'valid'):
	print('\ndata type =', data)
	
	path = '../HW5_data/TrimmedVideos/label/gt_' + data + '.csv'
	od = reader.getVideoList(path)
	print('len(od) =', len(od))
	print("len(od['Video_name']) =", len(od['Video_name']))
	print("len(od['Action_labels']) =", len(od['Action_labels']))
	print("len(od['Action_labels'] =", len(od['Action_labels']))

	path = '../HW5_data/TrimmedVideos/video/' + data + '/'
	num = len(od['Video_name'])
	
	print('num of videos =', num)
	videos = np.empty([num, 4, 240, 320, 3], np.uint8)
	labels = np.zeros([num, ], np.uint8)
	
	for i in range(num):
		#'Action_labels', 'Nouns', 'End_times', 'Start_times', 'Video_category', 'Video_index', 'Video_name'
		#readShortVideo(path, video_category, video_name, downsample_factor=12, rescale_factor=1)
		if i % 100 == 0 and i > 0: print(i)
		df = int(((int(od['End_times'][i]) - int(od['Start_times'][i]))/35.0 - 1)/3.0)
		video = reader.readShortVideo(path, od['Video_category'][i], od['Video_name'][i], downsample_factor=df, rescale_factor=1)
		print('video =', video.shape)
		count += video.shape[0]
		
		assert video.shape[0] >= 4
		videos[i] = video[:4]

		labels[i] = od['Action_labels'][i]
		leng_idx[i] = video.shape[0]
		
	print('videos.shape =', videos.shape)
	print('labels.shape =', labels.shape)
	print('labels =', labels[:13])
	
	return videos, labels, leng_idx


def import_trimmed(data = 'valid'):
	print('\ndata type =', data)
	
	path = '../HW5_data/TrimmedVideos/label/gt_' + data + '.csv'
	od = reader.getVideoList(path)
	print('len(od) =', len(od))
	print("len(od['Video_name']) =", len(od['Video_name']))
	print("len(od['Action_labels']) =", len(od['Action_labels']))
	print("len(od['Action_labels'] =", len(od['Action_labels']))

	path = '../HW5_data/TrimmedVideos/video/' + data + '/'
	num = len(od['Video_name'])
	
	print('num of videos =', num)
	videos = np.empty([4, 240, 320, 3], np.uint8)
	labels = np.zeros([num, ], np.uint8)
	df = 12
	count = 0
	leng_idx = np.zeros([num, ], np.uint32)
	for i in range(num):
		if i % 100 == 0 and i > 0: print(i)
		
		video = reader.readShortVideo(path, od['Video_category'][i], od['Video_name'][i], downsample_factor=df, rescale_factor=1)
		#print('video =', video.shape)
		if i == 0: videos = video
		else: videos = np.concatenate([videos, video])

		count += video.shape[0]
		labels[i] = od['Action_labels'][i]
		leng_idx[i] = video.shape[0]
		
	print("count =", count, np.sum(leng_idx))
	print('videos.shape =', videos.shape)
	print('labels.shape =', labels.shape)
	print('labels =', labels[:13])
	
	return videos, labels, leng_idx

def import_test_trimmed():	
	path = sys.argv[2] + '/gt_valid.csv'
	od = reader.getVideoList(path)
	print('len(od) =', len(od))

	path = sys.argv[1]
	num = len(od['Video_name'])
	print('num of videos =', num)
	
	df = 12
	count = 0
	leng_idx = np.zeros([num, ], np.uint32)
	for i in range(num):
		if i % 100 == 0 and i > 0: print(i)
		
		video = reader.readShortVideo(path, od['Video_category'][i], od['Video_name'][i], downsample_factor=df, rescale_factor=1)
		if i == 0: videos = video
		else: videos = np.concatenate([videos, video])

		count += video.shape[0]
		leng_idx[i] = video.shape[0]
		
	print("count =", count, np.sum(leng_idx))
	print('videos.shape =', videos.shape)
	
	return videos, leng_idx

def import_test_groudtruth():
	path = sys.argv[2] + '/gt_valid.csv'
	od = reader.getVideoList(path)
	print('len(od) =', len(od))

	path = sys.argv[1]
	num = len(od['Video_name'])
	print('num of videos =', num)

	labels = np.zeros([num, ], np.uint8)
	
	for i in range(num):
		if i % 100 == 0 and i > 0: print(i)
		labels[i] = od['Action_labels'][i]
		
	print('labels.shape =', labels.shape)
	print('labels =', labels[:13])
	
	return labels
