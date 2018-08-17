# coding: utf-8

import numpy as np
import tensorflow as tf
import fileinput
import matplotlib.pyplot as plt
import pylab
import wave, os, glob
import math
from scipy import signal
from scipy.io import wavfile
from keras.preprocessing import sequence


def rectified_noisy_label_V1(data, label):
	label = label.astype(np.float32)
	a, e, n, p, r = separa_fau_five(data, label)
	a_index = A_index(data, label)
	e_index = E_index(data, label)
	n_index = N_index(data, label)
	p_index = P_index(data, label)
	r_index = R_index(data, label)
	for i in range(5):
		if i == 0:
			merge = [e,n,p,r]
			for j in range(a.shape[0]):
				avg_dis = np.zeros(4)
				for k in range(4):
					compare_data = merge[k]
					dis = []
					for m in range(compare_data.shape[0]):
						euclidean_dis = distance.euclidean(a[j], compare_data[m])
						dis.append(euclidean_dis)
					avg_dis[k] = np.mean(dis, axis=0)
				cm = np.prod(avg_dis) # common multiple
				inv_ratio = np.array([cm/avg_dis[0], cm/avg_dis[1], cm/avg_dis[2], cm/avg_dis[3]])
				inv_sum = np.sum(inv_ratio)
				#print ('inv_ratio: {} , inv_sum: {}'.format(inv_ratio, inv_sum))
				rectified_label = np.array([0.8, 0.2*(inv_ratio[0]/inv_sum), 0.2*(inv_ratio[1]/inv_sum), 0.2*(inv_ratio[2]/inv_sum), 0.2*(inv_ratio[3]/inv_sum)], dtype=np.float32)
				#print ('rectified_label: {}'.format(rectified_label))
				label[math.floor(a_index[j])] = rectified_label
		elif i == 1:
			merge = [a,n,p,r]
			for j in range(e.shape[0]):
				avg_dis = np.zeros(4)
				for k in range(4):
					compare_data = merge[k]
					dis = []
					for m in range(compare_data.shape[0]):
						euclidean_dis = distance.euclidean(e[j], compare_data[m])
						dis.append(euclidean_dis)
					avg_dis[k] = np.mean(dis, axis=0)
				cm = np.prod(avg_dis)# common multiple
				inv_ratio = np.array([cm/avg_dis[0], cm/avg_dis[1], cm/avg_dis[2], cm/avg_dis[3]])
				inv_sum = np.sum(inv_ratio)
				rectified_label = np.array([0.2*(inv_ratio[0]/inv_sum), 0.8, 0.2*(inv_ratio[1]/inv_sum), 0.2*(inv_ratio[2]/inv_sum), 0.2*(inv_ratio[3]/inv_sum)], dtype=np.float32)
				label[math.floor(e_index[j])] = rectified_label
		elif i == 2:
			merge = [a,e,p,r]
			for j in range(n.shape[0]):
				avg_dis = np.zeros(4)
				for k in range(4):
					compare_data = merge[k]
					dis = []
					for m in range(compare_data.shape[0]):
						euclidean_dis = distance.euclidean(n[j], compare_data[m])
						dis.append(euclidean_dis)
					avg_dis[k] = np.mean(dis, axis=0)
				cm = np.prod(avg_dis)
				inv_ratio = np.array([cm/avg_dis[0], cm/avg_dis[1], cm/avg_dis[2], cm/avg_dis[3]])
				inv_sum = np.sum(inv_ratio)
				rectified_label = np.array([0.2*(inv_ratio[0]/inv_sum), 0.2*(inv_ratio[1]/inv_sum), 0.8, 0.2*(inv_ratio[2]/inv_sum), 0.2*(inv_ratio[3]/inv_sum)], dtype=np.float32)
				label[math.floor(n_index[j])] = rectified_label
		elif i == 3:
			merge = [a,e,n,r]
			for j in range(p.shape[0]):
				avg_dis = np.zeros(4)
				for k in range(4):
					compare_data = merge[k]
					dis = []
					for m in range(compare_data.shape[0]):
						euclidean_dis = distance.euclidean(p[j], compare_data[m])
						dis.append(euclidean_dis)
					avg_dis[k] = np.mean(dis, axis=0)
				cm = np.prod(avg_dis)
				inv_ratio = np.array([cm/avg_dis[0], cm/avg_dis[1], cm/avg_dis[2], cm/avg_dis[3]])
				inv_sum = np.sum(inv_ratio)
				rectified_label = np.array([0.2*(inv_ratio[0]/inv_sum), 0.2*(inv_ratio[1]/inv_sum), 0.2*(inv_ratio[2]/inv_sum), 0.8, 0.2*(inv_ratio[3]/inv_sum)], dtype=np.float32)
				label[math.floor(p_index[j])] = rectified_label
		elif i == 4:
			merge = [a,e,n,p]
			for j in range(r.shape[0]):
				avg_dis = np.zeros(4)
				for k in range(4):
					compare_data = merge[k]
					dis = []
					for m in range(compare_data.shape[0]):
						euclidean_dis = distance.euclidean(r[j], compare_data[m])
						dis.append(euclidean_dis)
					avg_dis[k] = np.mean(dis, axis=0)
				cm = np.prod(avg_dis)
				inv_ratio = np.array([cm/avg_dis[0], cm/avg_dis[1], cm/avg_dis[2], cm/avg_dis[3]])
				inv_sum = np.sum(inv_ratio)
				rectified_label = np.array([0.5*(inv_ratio[0]/inv_sum), 0.5*(inv_ratio[1]/inv_sum), 0.5*(inv_ratio[2]/inv_sum), 0.5*(inv_ratio[3]/inv_sum), 0.5], dtype=np.float32)
				label[math.floor(r_index[j])] = rectified_label
	return label
		

def rectified_noisy_label_V2(data, label):
	label = label.astype(np.float32)
	a, e, n, p, r = separa_fau_five(data, label)
	a_index = A_index(data, label)
	e_index = E_index(data, label)
	n_index = N_index(data, label)
	p_index = P_index(data, label)
	r_index = R_index(data, label)
	for i in range(5):# for each class
		if i == 0:
			merge = [a,e,n,p,r]
			for j in range(a.shape[0]):
				avg_dis = np.zeros(5)
				for k in range(5):
					compare_data = merge[k]
					dis = []
					for m in range(compare_data.shape[0]):
						if not np.array_equal(a[j], compare_data[m]):
							euclidean_dis = cosine_similarity([a[j]], [compare_data[m]])
							dis.append(euclidean_dis)
					avg_dis[k] = np.mean(dis, axis=0)
				cm = np.prod(avg_dis)# common multiple
				inv_ratio = np.array([cm/avg_dis[0], cm/avg_dis[1], cm/avg_dis[2], cm/avg_dis[3], cm/avg_dis[4]])
				inv_sum = np.sum(inv_ratio)
				rectified_label = np.array([1*(inv_ratio[0]/inv_sum), 1*(inv_ratio[1]/inv_sum), 1*(inv_ratio[2]/inv_sum), 1*(inv_ratio[3]/inv_sum), 1*(inv_ratio[4]/inv_sum)], dtype=np.float32)
				label[math.floor(a_index[j])] = rectified_label
		elif i == 1:
			merge = [a,e,n,p,r]
			for j in range(e.shape[0]):
				avg_dis = np.zeros(5)
				for k in range(5):
					compare_data = merge[k]
					dis = []
					for m in range(compare_data.shape[0]):
						if not np.array_equal(e[j], compare_data[m]):
							euclidean_dis = cosine_similarity([e[j]], [compare_data[m]])
							dis.append(euclidean_dis)
					avg_dis[k] = np.mean(dis, axis=0)
				cm = np.prod(avg_dis) # common multiple
				inv_ratio = np.array([cm/avg_dis[0], cm/avg_dis[1], cm/avg_dis[2], cm/avg_dis[3], cm/avg_dis[4]])
				inv_sum = np.sum(inv_ratio)
				rectified_label = np.array([1*(inv_ratio[0]/inv_sum), 1*(inv_ratio[1]/inv_sum), 1*(inv_ratio[2]/inv_sum), 1*(inv_ratio[3]/inv_sum), 1*(inv_ratio[4]/inv_sum)], dtype=np.float32)
				label[math.floor(e_index[j])] = rectified_label
		elif i == 2:
			merge = [a,e,n,p,r]
			for j in range(n.shape[0]):
				avg_dis = np.zeros(5)
				for k in range(5):
					compare_data = merge[k]
					dis = []
					for m in range(compare_data.shape[0]):
						if not np.array_equal(n[j], compare_data[m]):
							euclidean_dis = cosine_similarity([n[j]], [compare_data[m]])
							dis.append(euclidean_dis)
					avg_dis[k] = np.mean(dis, axis=0)
				cm = np.prod(avg_dis)
				inv_ratio = np.array([cm/avg_dis[0], cm/avg_dis[1], cm/avg_dis[2], cm/avg_dis[3], cm/avg_dis[4]])
				inv_sum = np.sum(inv_ratio)
				rectified_label = np.array([1*(inv_ratio[0]/inv_sum), 1*(inv_ratio[1]/inv_sum), 1*(inv_ratio[2]/inv_sum), 1*(inv_ratio[3]/inv_sum), 1*(inv_ratio[4]/inv_sum)], dtype=np.float32)
				label[math.floor(n_index[j])] = rectified_label
		elif i == 3:
			merge = [a,e,n,p,r]
			for j in range(p.shape[0]):
				avg_dis = np.zeros(5)
				for k in range(5):
					compare_data = merge[k]
					dis = []
					for m in range(compare_data.shape[0]):
						if not np.array_equal(p[j], compare_data[m]):
							euclidean_dis = cosine_similarity([p[j]], [compare_data[m]])
							dis.append(euclidean_dis)
					avg_dis[k] = np.mean(dis, axis=0)
				cm = np.prod(avg_dis)
				inv_ratio = np.array([cm/avg_dis[0], cm/avg_dis[1], cm/avg_dis[2], cm/avg_dis[3], cm/avg_dis[4]])
				inv_sum = np.sum(inv_ratio)
				rectified_label = np.array([1*(inv_ratio[0]/inv_sum), 1*(inv_ratio[1]/inv_sum), 1*(inv_ratio[2]/inv_sum), 1*(inv_ratio[3]/inv_sum), 1*(inv_ratio[4]/inv_sum)], dtype=np.float32)
				label[math.floor(p_index[j])] = rectified_label
		elif i == 4:
			merge = [a,e,n,p,r]
			for j in range(r.shape[0]):
				avg_dis = np.zeros(5)
				for k in range(5):
					compare_data = merge[k]
					dis = []
					for m in range(compare_data.shape[0]):
						if not np.array_equal(r[j], compare_data[m]):
							euclidean_dis = cosine_similarity([r[j]], [compare_data[m]])
							dis.append(euclidean_dis)
					avg_dis[k] = np.mean(dis, axis=0)
				cm = np.prod(avg_dis)
				inv_ratio = np.array([cm/avg_dis[0], cm/avg_dis[1], cm/avg_dis[2], cm/avg_dis[3], cm/avg_dis[4]])
				inv_sum = np.sum(inv_ratio)
				rectified_label = np.array([1*(inv_ratio[0]/inv_sum), 1*(inv_ratio[1]/inv_sum), 1*(inv_ratio[2]/inv_sum), 1*(inv_ratio[3]/inv_sum), 1*(inv_ratio[4]/inv_sum)], dtype=np.float32)
				label[math.floor(r_index[j])] = rectified_label
	#for i in range(label.shape[0]):
		#print (label[i])
	return label



def interpolation(inputs, size):
	return tf.image.resize_images(inputs, size, method=tf.image.ResizeMethod.BILINEAR)

def _to_tensor(x, dtype):
	x = tf.convert_to_tensor(x)
	if x.dtype != dtype:
			x = tf.cast(x, dtype)
	return x

def weighted_loss(r):
	def new_loss(y_true, y_pred):
		_EPSILON = 10e-8
		y_pred /= tf.reduce_sum(y_pred, reduction_indices=len(y_pred.get_shape()) - 1, keep_dims=True)
		# manual computation of crossentropy
		epsilon = _to_tensor(_EPSILON, y_pred.dtype.base_dtype)
		y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
		
		#y_true = tf.cast(y_true, tf.float64)
		rr = tf.cast(r, tf.float32)
		new_R = tf.matmul(rr, y_true, transpose_b=True)
		
		return - new_R * tf.reduce_sum(y_true * tf.log(y_pred), reduction_indices=len(y_pred.get_shape()) - 1)

	return new_loss


# read data in 3D form directly
def readFile(filename, isSP=False ) :
	f = open(filename, 'r')
	
	# read header
	line = f.readline()
	while line != "@data\n":
		line = f.readline()
	
	line = f.readline()  #space line
	
	#read data
	twoD_data = [] # store each frame of one data (shape: [timesteps, features])
	threeD_data = [] # store each data (shape: [data size, timesteps, features])
	label = []
	L = [0,0,0,0,0]
	seq_label = [0,0,0,0,0]
	each_wav_length = []
	wav_length = 0
	same = True # compare the previous filename and current filename
	previousFileName = ''# store the previous filename
	while True:
			line = f.readline()
			if line == '' :  # EOF
					break
			
			#data processing
			temp = []
			for numstr in line.split(",") :
				if numstr :
					try :
						numFl = float(numstr)
						temp.append(numFl)
					except ValueError as e :
						if not isSP :
								numstr = numstr[:-1]  #skip '\n'
						else :
								numstr = numstr[1]  #get class 

						if numstr[0] == '\'':
							if previousFileName == numstr:
								same = True
							previousFileName = numstr
						elif numstr == 'Anger' or numstr[0] == 'A' :##mark as 0
								L = [1,0,0,0,0]
						elif numstr == 'Emphatic' or numstr[0] == 'E' :##mark as 1
								L = [0,1,0,0,0]
						elif numstr == 'Neutral' or numstr[0] == 'N' :##mark as 2
								L = [0,0,1,0,0]
						elif numstr == 'Positive' or numstr[0] == 'P' :##mark as 3
								L = [0,0,0,1,0]
						elif numstr == 'Rest' or numstr[0] == 'R' :##mark as 4
								L = [0,0,0,0,1]
				# end of if
			if same == True: 
				twoD_data.append(temp) # end of for
				same = False
				wav_length += 1
				seq_label = L
			else:
				threeD_data.append(twoD_data)
				each_wav_length.append(wav_length)
				label.append(seq_label)
				wav_length = 0
				wav_length += 1
				twoD_data = []
				twoD_data.append(temp)
	# end of while
	f.close()
	threeD_data.append(twoD_data)
	each_wav_length.append(wav_length)
	label.append(seq_label)

	threeD_data = np.asarray(threeD_data)
	each_wav_length = np.asarray(each_wav_length)
	label = np.asarray(label)
	return threeD_data, label, each_wav_length

# read data in 2D form
def read_2D_data(filename, isSP=False) :
	f = open(filename, 'r')
	
	# read header
	line = f.readline()
	while line != "@data\n":
		line = f.readline()
	
	line = f.readline()  #space line
	
	#read data
	data = []
	label = []
	size = [0, 0, 0, 0, 0]
	L = [0,0,0,0,0]	
	data_info = []# store the data information
	while True:
			line = f.readline()
			if line == '' :  # EOF
					break
			
			#data processing
			temp = []
			
			for numstr in line.split(",") :
				if numstr :
					try :
						numFl = float(numstr)
						temp.append(numFl)
					except ValueError as e :

						if not isSP :
							numstr = numstr[:-1]  #skip '\n'
						else :
							numstr = numstr[1]  #get class (supervector only)
						if numstr[0] == '\'':
							data_info.append(numstr)
						if numstr == 'Anger' or numstr[0] == 'A' :##mark as 0
							L = [1,0,0,0,0]
							size[0] += 1
							label.append(L)  # get label
						if numstr == 'Emphatic' or numstr[0] == 'E' :##mark as 1
							L = [0,1,0,0,0]
							size[1] += 1
							label.append(L)  # get label
						if numstr == 'Neutral' or numstr[0] == 'N' :##mark as 2
							L = [0,0,1,0,0]
							size[2] += 1
							label.append(L)  # get label
						if numstr == 'Positive' or numstr[0] == 'P' :##mark as 3
							L = [0,0,0,1,0]
							size[3] += 1
							label.append(L)  # get label
						if numstr == 'Rest' or numstr[0] == 'R' :##mark as 4
							L = [0,0,0,0,1]
							size[4] += 1
							label.append(L)  # get label
			data.append(temp)  # get data
	
	f.close()
	
	data = np.asarray(data)
	label = np.asarray(label)
	data_info = np.asarray(data_info)
	size = np.asarray(size)
	return data, label, size, data_info

# convert 2D data to 3D
def convert_to_3Ddata(data, filename) :	
	twoD_data = [] # store each frame of one data (shape: [timesteps, features])
	threeD_data = [] # store each data (shape: [data size, timesteps, features])
	for i in range(filename.shape[0]):
		if i == 0:
			twoD_data.append(data[i])
		elif filename[i] == filename[i-1]:
			twoD_data.append(data[i])
			if i+1 == filename.shape[0]:
				threeD_data.append(twoD_data)
		else:
			threeD_data.append(twoD_data)
			twoD_data = []
			twoD_data.append(data[i])

	threeD_data = np.asarray(threeD_data)

	return threeD_data
def A_index(data, label):
	index = np.zeros(881)
	j = 0
	for i in range(data.shape[0]):
		if np.argmax(label[i]) == 0:
			index[j] = i
			j+=1
	return index
def E_index(data, label):
	index = np.zeros(2093)
	j = 0
	for i in range(data.shape[0]):
		if np.argmax(label[i]) == 1:
			index[j] = i
			j+=1
	return index

def N_index(data, label):
	index = np.zeros(5590)
	j = 0
	for i in range(data.shape[0]):
		if np.argmax(label[i]) == 2:
			index[j] = i
			j+=1
	return index

def P_index(data, label):
	index = np.zeros(674)
	j = 0
	for i in range(data.shape[0]):
		if np.argmax(label[i]) == 3:
			index[j] = i
			j+=1
	return index

def R_index(data, label):
	index = np.zeros(721)
	j = 0
	for i in range(data.shape[0]):
		if np.argmax(label[i])==4:
			index[j] = i
			j+=1
	return index

def separa_fau_five(data, label):
	a = []
	e = []
	n = []
	p = []
	r = []
	for i in range(data.shape[0]):
		if np.argmax(label[i]) == 0:
			a.append(data[i])
		if np.argmax(label[i]) == 1:
			e.append(data[i])
		if np.argmax(label[i]) == 2:
			n.append(data[i])
		if np.argmax(label[i]) == 3:
			p.append(data[i])
		if np.argmax(label[i]) == 4:
			r.append(data[i])
	a = np.asarray(a)
	e = np.asarray(e)
	n = np.asarray(n)
	p = np.asarray(p)
	r = np.asarray(r)
	return a,e,n,p,r

def combine_five(a,b,c,d,e):
		result = np.concatenate((a,b),axis=0)
		result = np.concatenate((result,c),axis=0)
		result = np.concatenate((result,d),axis=0)
		result = np.concatenate((result,e),axis=0)
		return result


def normalize_by_each_frame(data, lower=-1.0, upper=1.0):
		for i in range(data.shape[0]) :
			#_max = max(data[i,:])
			#_min = min(data[i,:])
			mean = np.mean(data[i, :])
			stddev = np.std(data[i, :])
			for j in range(data.shape[1]) :
				data[i,j] = (data[i,j] - mean) / stddev
				#data[i,k] = (((data[i][k] - _min) * (upper - lower)) / (_max - _min)) + lower
		return data

def normalize_by_each_feature(data, lower=0.0, upper=1.0):
		epsilon = 1e-10
		for i in range(data.shape[1]) :
			_max = max(data[:,i])
			_min = min(data[:,i])
			#mean = np.mean(data[:,i])
			#stddev = np.std(data[:,i])
			for k in range(data.shape[0]) :
				#data[k,i] = (data[k,i] - mean) / max(stddev, epsilon)
				data[k,i] = (((data[k][i] - _min) * (upper - lower)) / (_max - _min)) + lower
		return data
# one virtual speaker normalization
def normalize_by_whole_set(data, lower=-1.0, upper=1.0):
		#_min = np.amin(data)
		#_max = np.amax(data)
		mean = np.mean(data)
		stddev = np.std(data)
		data = np.asarray(data)
		for i in range(data.shape[0]):
			for j in range(data.shape[1]):
				data[i,j] = (data[i,j] - mean) / stddev
				#data[i][j] = (((data[i][j] - _min) * (upper - lower)) / (_max - _min)) + lower

		return data

def normalize_by_each_seq_data(data, data_info):
	_EPSILON = 10e-7
	temp = []
	current_data = ''
	next_data = ''
	seq_nor_data = []
	frame_index = 0
	while True:
		if current_data == next_data:
			temp.append(data[frame_index])
		else:
			temp = np.asarray(temp)
			for col in range(temp.shape[1]):
				mean = np.mean(temp[:,col])
				stddev = np.std(temp[:,col])
				for row in range(temp.shape[0]):
					temp[row, col] = (temp[row, col] - mean) / (stddev+_EPSILON)
			for j in range(len(temp)):
				seq_nor_data.append(temp[j])
			if frame_index+1 == data_info.shape[0]:
				break
			temp = []
			temp.append(data[frame_index])
		current_data = data_info[frame_index] 
		if frame_index+1 < data_info.shape[0]:
			frame_index += 1
			next_data = data_info[frame_index]
		else:
			next_data = 'last'
	return seq_nor_data

def speaker_normalization(data, data_info, lower=-1.0, upper=1.0):
	_EPSILON = 10e-7
	temp = []
	current_speaker = ''
	next_speaker = ''
	sn_data = []
	frame_index = 0
	while True:
		if current_speaker == next_speaker:
			temp.append(data[frame_index])
		else:
			temp = np.asarray(temp)
			for col in range(temp.shape[1]):
				mean = np.mean(temp[:,col])
				stddev = np.std(temp[:,col])
				for row in range(temp.shape[0]):
					temp[row, col] = (temp[row, col] - mean) / (stddev+_EPSILON)
			for j in range(len(temp)):
				sn_data.append(temp[j])
			if frame_index+1 == data_info.shape[0]:
				break
			temp = []
			temp.append(data[frame_index])
		current_speaker = data_info[frame_index][-15] + data_info[frame_index][-14]
		if frame_index+1 < data_info.shape[0]:
			frame_index+=1
			next_speaker = data_info[frame_index][-15] + data_info[frame_index][-14]
		else:
			next_speaker = 'last'
	sn_data = np.asarray(sn_data)
	return sn_data

def convert_to_onehot(label):
	onehot=[]
	for i in label:
		if i == 0:
			L = [1,0,0,0,0]
		elif i == 1:
			L = [0,1,0,0,0]
		elif i == 2:
			L = [0,0,1,0,0]
		elif i == 3:
			L = [0,0,0,1,0]
		elif i == 4:
			L = [0,0,0,0,1]
		onehot.append(L)

	return onehot

def convert_to_non_onehot(label):
	new_label = []
	for i in range(label.shape[0]):
		if np.array_equal(label[i], [1,0,0,0,0]):
			new_label.append(0)
		elif np.array_equal(label[i], [0,1,0,0,0]):
			new_label.append(1)
		elif np.array_equal(label[i], [0,0,1,0,0]):
			new_label.append(2)
		elif np.array_equal(label[i], [0,0,0,1,0]):
			new_label.append(3)
		elif np.array_equal(label[i], [0,0,0,0,1]):
			new_label.append(4)
	new_label = np.asarray(new_label)
	return new_label

def add_label(filename):
	file = fileinput.input(filename, inplace=True)
	for line in file:
		data = line.split(',')
		data[0] = ''
		if 'A' in data[0]:
			line = line.replace('unknown', 'Anger')
			print (line, end='')
		elif 'E' in data[0]:
			line = line.replace('unknown', 'Emphatic')
			print (line, end='')
		elif 'N' in data[0]:
			line = line.replace('unknown', 'Neutral')
			print (line, end='')
		elif 'P' in data[0]:
			line = line.replace('unknown', 'Positive')
			print (line, end='')
		elif 'R' in data[0]:
			line = line.replace('unknown', 'Rest')
			print (line, end='')
		else:
			print (line, end='')
	file.close()



def show_confusion_matrix(predict, true_label, size):
	num_class = true_label.shape[1]
	AC = np.zeros(num_class)# store the classification result of each class
	AC_WA = 0.0
	AC_UA = 0.0
	confusion_matrix = np.zeros((num_class, num_class))# store the confusion matrix

	for i in range(predict.shape[0]):
		predict_class = np.argmax(predict[i])
		true_class = np.argmax(true_label[i])
		confusion_matrix[true_class][predict_class] += 1
	for i in range(num_class):
		AC[i] = confusion_matrix[i][i] / size[i]

	print("-------------Classification results-------------")
	print ('A: {0} , E: {1} , N: {2} , P:{3} , R: {4}'.format(size[0],size[1],size[2],size[3],size[4]))
	AC_UA = np.mean(AC)
	AC_WA = (confusion_matrix[0][0] + confusion_matrix[1][1] + confusion_matrix[2][2] + confusion_matrix[3][3] + confusion_matrix[4][4] ) / predict.shape[0]
	print('      A      E      N      P      R')
	print('A  %4d   %4d   %4d   %4d   %4d    ' % (confusion_matrix[0][0], confusion_matrix[0][1], confusion_matrix[0][2], confusion_matrix[0][3], confusion_matrix[0][4]))
	print('E  %4d   %4d   %4d   %4d   %4d    ' % (confusion_matrix[1][0], confusion_matrix[1][1], confusion_matrix[1][2], confusion_matrix[1][3], confusion_matrix[1][4]))
	print('N  %4d   %4d   %4d   %4d   %4d    ' % (confusion_matrix[2][0], confusion_matrix[2][1], confusion_matrix[2][2], confusion_matrix[2][3], confusion_matrix[2][4]))
	print('P  %4d   %4d   %4d   %4d   %4d    ' % (confusion_matrix[3][0], confusion_matrix[3][1], confusion_matrix[3][2], confusion_matrix[3][3], confusion_matrix[3][4]))
	print('R  %4d   %4d   %4d   %4d   %4d    ' % (confusion_matrix[4][0], confusion_matrix[4][1], confusion_matrix[4][2], confusion_matrix[4][3], confusion_matrix[4][4]))
	print('\nA: %f    E: %f     N: %f     P: %f     R: %f\n' % (AC[0]*100, AC[1]*100, AC[2]*100, AC[3]*100, AC[4]*100))
	print("WA: {}".format(AC_WA*100))
	print("UA: {}".format(AC_UA*100))
	print("------------------------------------------------")

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference



# lower, upper = 0.0, 1.0
# mu, sigma = 0, 0.05
# normal_attention = np.zeros(2453)
# lbound, rbound = -3, 3
# gap = (rbound-lbound)/2453
# x = lbound
# i = 0
# while x <= rbound:
# 	normal_attention[i] = (1/(np.sqrt(2*np.pi*np.square(sigma)))) * np.exp(-(np.square(x-mu))/(2*np.square(sigma)))
# 	i += 1
# 	x += gap

# _max = np.amax(normal_attention)
# _min = np.amin(normal_attention)
# for i in range(normal_attention.shape[0]):
# 	normal_attention[i] = (((normal_attention[i] - _min) * (upper - lower)) / (_max - _min)) + lower

# _sum = np.sum(normal_attention)
# for i in range(normal_attention.shape[0]):
# 	normal_attention[i] = normal_attention[i] / _sum
# #normal_attention = softmax(normal_attention)
# x = np.arange(0, 2453)
# plt.plot(x, normal_attention)
# # count, bins, ignored = plt.hist(normal, 100, density=True)
# # plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
# #          np.exp( - (bins - mu)**2 / (2 * sigma**2) ), linewidth=2, color='r')
# plt.show()
'''
train_gender_info = {}
test_gender_info = {}
with open('gender.txt', 'r') as f:
	for line in f:
		if line[0] == 'M': # test
			temp = line.split(' ')
			id_no = temp[0][5]+temp[0][6]# 
			temp[1] = temp[1].strip('\n')# remove the '\n' character
			if temp[1] == 'm': # male
				temp[1] = [1,0]
			elif temp[1] == 'f': # female
				temp[1] = [0,1]
			test_gender_info.update({id_no: temp[1]})
		elif line[0] == 'O': # train
			temp = line.split(' ')
			id_no = temp[0][4]+temp[0][5]
			temp[1] = temp[1].strip('\n')# remove the '\n' character
			if temp[1] == 'm': # male
				temp[1] = [1,0]
			elif temp[1] == 'f':# female
				temp[1] = [0,1]
			train_gender_info.update({id_no: temp[1]})

print (train_gender_info)
print (len(train_gender_info))
print ('=======================================================================')
print (test_gender_info)
print (len(test_gender_info))
print ('=======================================================================')

train, train_label, _, train_info = read_2D_data('./data/fau_train_with_info.arff')
test, test_label, _, test_info = read_2D_data('./data/fau_test_with_info.arff')

train_gender_label = []
for i in range(train_info.shape[0]):
	wav_filename = train_info[i].split('/')[7]
	id_no = wav_filename[4] + wav_filename[5]
	train_gender_label.append(train_gender_info[id_no])

test_gender_label = []
for i in range(test_info.shape[0]):
	wav_filename = test_info[i].split('/')[7]
	id_no = wav_filename[5] + wav_filename[6]
	test_gender_label.append(test_gender_info[id_no])

train_gender_label = np.asarray(train_gender_label)
test_gender_label = np.asarray(test_gender_label)
print (train_gender_label.shape , test_gender_label.shape)
np.save('fau_train_gender_label.npy', train_gender_label)
np.save('fau_test_gender_label.npy', test_gender_label)
'''

'''
_EPSILON = 10e-7
train, train_label, _, train_info = read_2D_data('./data/fau_train_lld_delta.arff')
print ('Raw: {} , {}'.format(train.shape, train_label.shape))
train = speaker_normalization(train, train_info)
print ('After sn: {}'.format(train.shape))

temp = []
for i in range(train.shape[0]):
	if np.array_equal(train_label[i], [0,0,0,0,1]):
		temp.append(train[i])
temp = np.asarray(temp)
print (temp.shape)
for col in range(temp.shape[1]):
	mean = np.mean(temp[:,col])
	stddev = np.std(temp[:,col])
	for row in range(temp.shape[0]):
		temp[row, col] = (temp[row, col] - mean) / (stddev+_EPSILON)
j = 0
for i in range(train.shape[0]):
	if np.array_equal(train_label[i], [0,0,0,0,1]):
		train[i] = temp[j]
		j += 1

train = convert_to_3Ddata(train, train_info)
print ('Convert to 3D: {}'.format(train.shape))
train = sequence.pad_sequences(train, maxlen=2453)
print ('After padding: {}'.format(train.shape))
np.save('./data/fau_train_lld_delta_fea_sn_pad_2.npy', train)

print ('======================================================')

test, test_label, _, test_info = read_2D_data('./data/fau_test_lld_delta.arff')
print ('Raw: {} , {}'.format(test.shape, test_label.shape))
test = speaker_normalization(test, test_info)
print ('After sn: {}'.format(test.shape))

temp2 = []
for i in range(test.shape[0]):
	if np.array_equal(test_label[i], [0,0,0,0,1]):
		temp2.append(test[i])
temp2 = np.asarray(temp2)
print (temp2.shape)
for col in range(temp2.shape[1]):
	mean = np.mean(temp2[:,col])
	stddev = np.std(temp2[:,col])
	for row in range(temp2.shape[0]):
		temp2[row, col] = (temp2[row, col] - mean) / (stddev+_EPSILON)
j = 0
for i in range(test.shape[0]):
	if np.array_equal(test_label[i], [0,0,0,0,1]):
		test[i] = temp2[j]
		j += 1

test = convert_to_3Ddata(test, test_info)
print ('Convert to 3D: {}'.format(test.shape))
test = sequence.pad_sequences(test, maxlen=2453)
print ('After padding: {}'.format(test.shape))
np.save('./data/fau_test_lld_delta_fea_sn_pad_2.npy', test)

'''

