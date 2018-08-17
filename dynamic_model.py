import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import argparse

from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import *
from keras.layers import *
from keras.layers.merge import multiply, add, concatenate
from keras_contrib.layers import *

from keras.models import load_model
from keras.optimizers import *
from keras.layers.wrappers import TimeDistributed,Bidirectional
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from keras.utils import plot_model, print_summary
from keras.regularizers import *
from keras import backend as K
from utils import *


MODEL_ROOT_DIR = './dynamic_model/'
# Learning parameters
BATCH_SIZE = 100
TRAIN_EPOCHS = 10
LEARNING_RATE = 0.002
DECAY = 0.00001
MOMENTUM = 0.9
# Network parameters
CLASSES = 5 # A,E,N,P,R
DYNAMIC_FEATURES = 32
STATIC_FEATURES = 384
LSTM_UNITS = 60
ATTENTION_LAYER_UNITS = 50

def get_arguments():
	parser = argparse.ArgumentParser(description='Fau-Aibo with dynamic model.')
	parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
						help='How many train data to process at once. Default: ' + str(BATCH_SIZE) + '.')
	parser.add_argument('--model_root_dir', type=str, default=MODEL_ROOT_DIR,
						help='Root directory to place the trained model.')
	parser.add_argument('--load_model', type=str, default='',
						help='Whether to load existing model for test.')
	parser.add_argument('--train_epochs', type=int, default=TRAIN_EPOCHS,
						help='Number of training epochs. Default: ' + str(TRAIN_EPOCHS) + '.')
	parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE,
						help='Learning rate for training. Default: ' + str(LEARNING_RATE) + '.')
	parser.add_argument('--decay', type=float, default=DECAY,
						help='Decay rate for training. Default: ' + str(DECAY) + '.')
	parser.add_argument('--momentum', type=float,default=MOMENTUM, 
						help='Momentum for training. Default: ' + str(MOMENTUM) + '.')
	parser.add_argument('--classes', type=int,default=CLASSES, 
						help='number of classes in RNN network.')
	parser.add_argument('--dynamic_features', type=int,default=DYNAMIC_FEATURES, 
						help='dimension of features of the dynamic model.')
	parser.add_argument('--static_features', type=int,default=STATIC_FEATURES, 
						help='dimension of features of the static model.')
	parser.add_argument('--normalize_mode',type=int,default=0,
						help='Choose the normalize mode (0: no,1: feature-wise,2: audio-wise,3: speaker-wise).')
	parser.add_argument('--lstm_units', type=int, default=LSTM_UNITS,
						help='The units of the LSTM layer.')
	parser.add_argument('--attention_layer_units', type=int, default=ATTENTION_LAYER_UNITS,
						help='The units of the attention layer.')
	return parser.parse_args()

def get_last_output(x) :
		return x[:,-1,:]


def train_RNN(train_data, train_label, train_size,test_data, test_label, test_size, max_length, bagging_model_path=''):
	# use LSTM model for dynamic model
	args = get_arguments()
	class_weight = {0: 1.1, 1: 0.5, 2: 0.2, 3: 1.5, 4: 1.4}
	data_weights = []
	data_weights.append([1.1, 0.5, 0.2, 1.5, 1.4])# original
	data_weights = np.asarray(data_weights)

	dynamic_input = Input(shape=[max_length, args.dynamic_features], dtype='float32', name='dynamic_input')

	lstm1 = LSTM(60, activation='tanh', return_sequences=True, recurrent_dropout=0.5, name='lstm1')(dynamic_input)
	
	
	#attention mechanism module (forward)
	attention_dense1_f = Dense(args.attention_layer_units, activation='tanh', use_bias=False, name='attention_dense1_f')(lstm1)
	attention_dense2_f = Dense(1, use_bias=False, name='attention_dense2_f')(attention_dense1_f)
	attention_flatten_f = Flatten()(attention_dense2_f)
	attention_softmax_f = Activation('softmax', name='attention_weights_f')(attention_flatten_f)
	attention_multiply_f = multiply([lstm1, attention_permute_f])
	attention_representation = Lambda(lambda xin: K.sum(xin, axis=1), name='attention_representation')(attention_multiply_f)# 60

	
	# classifier module
	#mean = Lambda(lambda xin: K.mean(xin, axis=1))(lstm2)
	output = Dense(args.classes, activation='softmax', name='output')(attention_representation)
	model = Model(inputs=dynamic_input, outputs=output)
	model.summary()

	if args.load_model == '':
		#training
		optimizer = Adam(lr=args.learning_rate, decay=args.decay)
		#model.compile(loss=weighted_loss(r=data_weights), optimizer=optimizer, metrics=['accuracy'])
		model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
		earlystopping = EarlyStopping(monitor='loss', min_delta=0, patience=0)
		checkpoint = ModelCheckpoint(filepath='dynamic_model/best_checkpoint.hdf5', monitor='val_loss', save_best_only=True)
		callbacks_list = [earlystopping, checkpoint]
		model.fit(x=train_data, y=train_label, batch_size=args.batch_size, epochs=args.train_epochs
							, callbacks=callbacks_list, verbose=2, class_weight=class_weight) 
		# save the model
		model.save_weights(args.model_root_dir + 'model.h5')
	else:
		print ('Load the model: {}'.format(args.load_model))
		model.load_weights(args.model_root_dir + args.load_model)
		print('==========================================================')
	#predict
	predict = model.predict(test_data)
	show_confusion_matrix(predict, test_label, test_size)

if __name__ == '__main__':
	args = get_arguments()
	print('==========================================================')
	# No normalize
	print ('Normalization mode: speaker-wise CMVN')
	
	train_data = np.load('./data/fau_train_lld_delta_fea_sn_pad.npy')
	test_data = np.load('./data/fau_test_lld_delta_fea_sn_pad.npy')
	

	train_label = np.load('./data/fau_train_label.npy')
	train_size = np.array([881, 2093, 5590, 674, 721])
	train_seq_length = np.load('./data/fau_train_length_lld_delta.npy')
	test_label = np.load('./data/fau_test_label.npy')
	test_size = np.array([611, 1508, 5377, 215, 546])
	test_seq_length = np.load('./data/fau_test_length_lld_delta.npy')
	max_length = max(np.amax(train_seq_length), np.amax(test_seq_length))


	print('==========================================================')
	print('Train data and label: {} , {}'.format(train_data.shape, train_label.shape))
	print('Test data and label: {} , {}'.format(test_data.shape, test_label.shape))
	print('==========================================================')

	
	train_RNN(train_data, train_label, train_size, test_data, test_label, test_size, max_length)


