import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import math
from utils import *
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD, Adam
from keras import backend as K

def second_max(x):
	x_sorted = sorted(x, reverse=True)
	return x[1]

def ensemble_train(train_data, train_label, test_data, test_label):
	input_dim = train_data.shape[1]
	learning_rate = 0.002
	decay = 0.00001
	momentum = 0.5
	batch_size = 100
	epoch = 20

	data_weights = []
	data_weights.append([1.1,0.5,0.2,1.5,1.4])
	data_weights = np.asarray(data_weights)

	model = Sequential()
	model.add(Dense(20, activation='sigmoid', input_shape=(input_dim,)))
	model.add(Dense(5, activation='softmax'))
	model.summary()
	adam = Adam(lr=learning_rate, decay=decay)
	model.compile(loss=weighted_loss(r=data_weights), optimizer=adam,  metrics=['accuracy'])
	model.fit(train_data, train_label, batch_size=batch_size, epochs=epoch, verbose=2)    

	predict = model.predict(test_data)
	return predict

if __name__ == '__main__':
	static_mlp_predict_test = np.load('./predict/mlp_2layers_predict_4615.npy')
	static_conv_predict_test = np.load('./predict/conv_ts_predict_4631.npy')
	dynamic_predict_test = np.load('./predict/dynamic_ts_predict_4717.npy')
	#static_add_predict = np.load('./predict/conv_mlp_add_ts_predict_4678.npy')
	test_label = np.load('./data/fau_test_label.npy')
	test_size = np.array([611, 1508, 5377, 215, 546])

	ensemble_predict = dynamic_predict_test*0.5 + static_conv_predict_test*0.1 + static_mlp_predict_test*0.4
	show_confusion_matrix(ensemble_predict, test_label, test_size)

	
	'''
	concatenate_predict_test = static_predict_test+dynamic_predict_test
	static_predict_train = np.load('./predict/static_train_predict.npy')
	dynamic_predict_train = np.load('./predict/dynamic_ts_train_predict.npy')
	concatenate_predict_train = static_predict_train+dynamic_predict_train

	train_label = np.load('./data/fau_train_label.npy')
	train_size = np.array([881, 2093, 5590, 674, 721])
	test_data, test_label, test_size, _ = read_2D_data('./data/CS_Ftest_nor.arff')

	print ('Train: {}'.format(concatenate_predict_train.shape))
	print ('Test: {}'.format(concatenate_predict_test.shape))
	final_predict = ensemble_train(concatenate_predict_train, train_label, concatenate_predict_test, test_label)
	show_confusion_matrix(final_predict, test_label, test_size)
	'''
	

	