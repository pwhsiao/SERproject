import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
from utils import *



if __name__ == '__main__':
	static_mlp_predict_test = np.load('./predict/mlp_2layers_predict_4615.npy')
	static_conv_predict_test = np.load('./predict/conv_ts_predict_4631.npy')
	dynamic_predict_test = np.load('./predict/dynamic_ts_predict_4717.npy')
	
	test_label = np.load('./data/fau_test_label.npy')
	test_size = np.array([611, 1508, 5377, 215, 546])

	ensemble_predict = dynamic_predict_test*0.5 + static_conv_predict_test*0.1 + static_mlp_predict_test*0.4
	show_confusion_matrix(ensemble_predict, test_label, test_size)

	
	

	