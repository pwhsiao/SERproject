import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import math
import numpy as np
from utils import *

def delete_padding_part(data, seq_length):
	no_padding_data = []
	for i in range(data.shape[0]):
		no_padding_start_pos = 2453 - seq_length[i]
		temp = data[i][no_padding_start_pos:]
		no_padding_data.append(temp)
	no_padding_data = np.asarray(no_padding_data)
	return no_padding_data

def avg_distribution(data, max_slot=100):
	seq_slot_value = []
	for i in range(data.shape[0]):# each sequence
		interval = len(data[i])/max_slot
		temp = []
		slot_count = 0
		j = 0.0
		while slot_count < max_slot:# each slot
			slot_sum = np.zeros(60)
			count = 0
			if math.floor(j) != math.floor(j+interval):
				for k in range(math.floor(j), math.floor(j+interval)):
					if k >= len(data[i]):
						break
					count += 1
					slot_sum += data[i][k]
			else:
				slot_sum = np.zeros(60)
			temp.append(slot_sum)
			slot_count += 1
			j += interval
		seq_slot_value.append(temp)
	seq_slot_value = np.asarray(seq_slot_value)
	# calculate the average by each column
	seq_slot_avg = []
	for i in range(seq_slot_value.shape[1]):
		col_sum = np.sum(seq_slot_value[:, i], axis=0)
		col_sum_avg = col_sum / (seq_slot_value.shape[0])
		seq_slot_avg.append(col_sum_avg)
	seq_slot_avg = np.asarray(seq_slot_avg)
	return seq_slot_avg


if __name__ == "__main__":
	
	A_seq_length = np.load('./data/test_length_A.npy')
	E_seq_length = np.load('./data/test_length_E.npy')
	N_seq_length = np.load('./data/test_length_N.npy')
	P_seq_length = np.load('./data/test_length_P.npy')
	R_seq_length = np.load('./data/test_length_R.npy')
	#attention_f = np.load('./attention_data/test_attention_weights_f.npy')
	attention_b = np.load('./attention_data/test_attention_weights_b.npy')
	#print ('Attention weights forward: {}'.format(attention_f.shape))
	print ('Attention weights backward: {}'.format(attention_b.shape))
	print('===============================================================')
	label = np.load('./data/fau_test_label.npy')
	attention_A, attention_E, attention_N, attention_P, attention_R = separa_fau_five(attention_b, label)
	# deleting the zero-padding part
	attention_A = delete_padding_part(attention_A, A_seq_length)
	attention_E = delete_padding_part(attention_E, E_seq_length)
	attention_N = delete_padding_part(attention_N, N_seq_length)
	attention_P = delete_padding_part(attention_P, P_seq_length)
	attention_R = delete_padding_part(attention_R, R_seq_length)
	
	print ('A : {}'.format(attention_A.shape))
	print ('E : {}'.format(attention_E.shape))
	print ('N : {}'.format(attention_N.shape))
	print ('P : {}'.format(attention_P.shape))
	print ('R : {}'.format(attention_R.shape))
	
	attention_A_avg = avg_distribution(attention_A)
	attention_E_avg = avg_distribution(attention_E)
	attention_N_avg = avg_distribution(attention_N)
	attention_P_avg = avg_distribution(attention_P)
	attention_R_avg = avg_distribution(attention_R)
	'''
	attention_A_avg = np.load('att_matrix_avg_A.npy')
	attention_E_avg = np.load('att_matrix_avg_E.npy')
	attention_N_avg = np.load('att_matrix_avg_N.npy')
	attention_P_avg = np.load('att_matrix_avg_P.npy')
	attention_R_avg = np.load('att_matrix_avg_R.npy')
	attention_A_avg_1D = np.mean(attention_A_avg, axis=1)
	attention_E_avg_1D = np.mean(attention_E_avg, axis=1)
	attention_N_avg_1D = np.mean(attention_N_avg, axis=1)
	attention_P_avg_1D = np.mean(attention_P_avg, axis=1)
	attention_R_avg_1D = np.mean(attention_R_avg, axis=1)
	print (attention_A_avg_1D.shape)
	print (attention_E_avg_1D.shape)
	print (attention_N_avg_1D.shape)
	print (attention_P_avg_1D.shape)
	print (attention_R_avg_1D.shape)
	'''
	red_patch = mpatches.Patch(color='red', label='Anger')
	green_patch = mpatches.Patch(color='green', label='Emphatic')
	blue_patch = mpatches.Patch(color='blue', label='Neutral')
	yellow_patch = mpatches.Patch(color='#F2F525', label='Positive')
	cyan_patch = mpatches.Patch(color='cyan', label='Rest')
	plt.legend(handles=[red_patch,green_patch,blue_patch,yellow_patch,cyan_patch])
	plt.plot(attention_A_avg, 'r', attention_E_avg, 'g', attention_N_avg, 'b',
			 attention_P_avg, '#F2F525', attention_R_avg, 'c', linewidth=1.5)
	#plt.show()
	filename = './attention_distribution_figure/blstm_test_attention_b_dis.png'
	plt.savefig(filename)
	#plt.clf()
	