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

from keras.optimizers import *
from keras.layers.wrappers import TimeDistributed,Bidirectional
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from keras.utils import plot_model
from keras.regularizers import *
from keras import backend as K
from utils import *


MODEL_ROOT_DIR = './dynamic_model/'

# Learning parameters
BATCH_SIZE = 128
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



def train(train_data, train_label, train_size,test_data, test_label, test_size, max_length):
	# use LSTM model for dynamic model
	args = get_arguments()
	class_weight = {0: 1.1, 1: 0.5, 2: 0.2, 3: 1.5, 4: 1.4}
	data_weights = []
	data_weights.append([1.1, 0.5, 0.2, 1.5, 1.4])# original

	data_weights = np.asarray(data_weights)
	# static model data
	mlp_train_data, _, _, _= read_2D_data('./data/CS_Ftrain_nor.arff')
	mlp_test_data, _, _, _ = read_2D_data('./data/CS_Ftest_nor.arff')

	cnn_train_data = np.reshape(mlp_train_data, (train_data.shape[0], 16, 12, 2))
	cnn_test_data = np.reshape(mlp_test_data, (test_data.shape[0], 16, 12, 2))

	dynamic_input = Input(shape=[max_length, args.dynamic_features], dtype='float32', name='dynamic_input')
	mlp_input = Input(shape=[args.static_features], dtype='float32', name='mlp_input')
	cnn_input = Input(shape=[16, 12, 2], dtype='float32', name='cnn_input')

	# LSTM module
	lstm1 = LSTM(args.lstm_units, activation='tanh', return_sequences=True, dropout=0.4, recurrent_dropout=0.2, name='lstm1')(dynamic_input)
	
	# attention mechanism module (forward)
	attention_dense1 = Dense(args.attention_layer_units, activation='tanh', use_bias=False, name='attention_dense1_f')(lstm1)
	attention_dense2 = Dense(1, use_bias=False, name='attention_dense2_f')(attention_dense1)
	attention_flatten = Flatten()(attention_dense2)
	attention_softmax = Activation('softmax', name='attention_weights_f')(attention_flatten)
	attention_repeat = RepeatVector(args.lstm_units)(attention_softmax)
	attention_permute = Permute([2, 1])(attention_repeat)
	attention_multiply = multiply([lstm1, attention_permute])
	attention_sum = Lambda(lambda xin: K.sum(xin, axis=1), name='attention_sum')(attention_multiply)# 60

	# MLP module
	mlp_output = Dense(units=60, activation='tanh', name='mlp_output')(mlp_input)# 60

	# CNN modeule
	conv1_1 =  Conv2D(filters=40,kernel_size=(1,1),strides=(2,2),padding='same', activation='tanh', kernel_initializer='TruncatedNormal', bias_initializer='TruncatedNormal', name='conv1_1')(cnn_input)
	conv1_2 =  Conv2D(filters=40,kernel_size=(3,3),strides=(2,2),padding='same', activation='tanh', kernel_initializer='TruncatedNormal', bias_initializer='TruncatedNormal', name='conv1_2')(cnn_input)
	conv1_3 =  Conv2D(filters=40,kernel_size=(5,5),strides=(2,2),padding='same', activation='tanh', kernel_initializer='TruncatedNormal', bias_initializer='TruncatedNormal', name='conv1_3')(cnn_input)
	conv1_4 =  Conv2D(filters=40,kernel_size=(7,7),strides=(2,2),padding='same', activation='tanh', kernel_initializer='TruncatedNormal', bias_initializer='TruncatedNormal', name='conv1_4')(cnn_input)
	conv1_5 =  Conv2D(filters=40,kernel_size=(9,9),strides=(2,2),padding='same', activation='tanh', kernel_initializer='TruncatedNormal', bias_initializer='TruncatedNormal', name='conv1_5')(cnn_input)
	conv1_maxout = maximum([conv1_1, conv1_2, conv1_3, conv1_4, conv1_5])
	#4x3x30
	conv2_1 =  Conv2D(filters=30,kernel_size=(1,1),strides=(1,1),padding='same', activation='tanh', kernel_initializer='TruncatedNormal', bias_initializer='TruncatedNormal', name='conv2_1')(conv1_maxout)
	conv2_2 =  Conv2D(filters=30,kernel_size=(3,3),strides=(1,1),padding='same', activation='tanh', kernel_initializer='TruncatedNormal', bias_initializer='TruncatedNormal', name='conv2_2')(conv1_maxout)
	conv2_3 =  Conv2D(filters=30,kernel_size=(5,5),strides=(1,1),padding='same', activation='tanh', kernel_initializer='TruncatedNormal', bias_initializer='TruncatedNormal', name='conv2_3')(conv1_maxout)
	conv2_4 =  Conv2D(filters=30,kernel_size=(7,7),strides=(1,1),padding='same', activation='tanh', kernel_initializer='TruncatedNormal', bias_initializer='TruncatedNormal', name='conv2_4')(conv1_maxout)
	conv2_maxout = maximum([conv2_1, conv2_2, conv2_3, conv2_4])
	#2x2x20
	conv3_1 = Conv2D(filters=20,kernel_size=(1,1),strides=(1,1),padding='same', activation='tanh', kernel_initializer='TruncatedNormal', bias_initializer='TruncatedNormal', name='conv3_1')(conv2_maxout)
	conv3_2 = Conv2D(filters=20,kernel_size=(3,3),strides=(1,1),padding='same', activation='tanh', kernel_initializer='TruncatedNormal', bias_initializer='TruncatedNormal', name='conv3_2')(conv2_maxout)
	conv3_3 = Conv2D(filters=20,kernel_size=(5,5),strides=(1,1),padding='same', activation='tanh', kernel_initializer='TruncatedNormal', bias_initializer='TruncatedNormal', name='conv3_3')(conv2_maxout)
	conv3_maxout = maximum([conv3_1, conv3_2, conv3_3])
	#1x1x10
	conv4_1 = Conv2D(filters=10, kernel_size=(1,1),strides=(1,1),padding='same', activation='tanh', kernel_initializer='TruncatedNormal', bias_initializer='TruncatedNormal', name='conv4_1')(conv3_maxout)
	conv4_2 = Conv2D(filters=10, kernel_size=(3,3),strides=(1,1),padding='same', activation='tanh', kernel_initializer='TruncatedNormal', bias_initializer='TruncatedNormal', name='conv4_2')(conv3_maxout)
	conv4_maxout = maximum([conv4_1, conv4_2])
	# attention module (input: 8x6x40)
	attention_pool_1 = MaxPooling2D(pool_size=(2,2), padding='same')(conv1_maxout)# 4x3x40
	attention_conv_1 = Conv2D(filters=30, kernel_size=(2,2), padding='same', use_bias=False, kernel_initializer='TruncatedNormal')(attention_pool_1)# 4x3x30
	attention_pool_2 = MaxPooling2D(pool_size=(2,2), padding='same')(attention_conv_1)#2x2x30
	attention_conv_2 = Conv2D(filters=20, kernel_size=(2,2), padding='same', use_bias=False, kernel_initializer='TruncatedNormal')(attention_pool_2)#2x2x20
	attention_interp_1 = Lambda(interpolation, arguments={'size': (4,3)})(attention_conv_2)# 4x3x20
	attention_conv_3 = Conv2D(filters=20, kernel_size=(2,2), padding='same', use_bias=False, kernel_initializer='TruncatedNormal')(attention_interp_1)#4x3x20
	attention_interp_2 = Lambda(interpolation, arguments={'size': (8,6)})(attention_conv_3)# 8x6x20
	attention_conv_4 = Conv2D(filters=10, kernel_size=(2,2), padding='same', use_bias=False, kernel_initializer='TruncatedNormal')(attention_interp_2)#8x6x10
	attention_weights = Activation('softmax', name='cnn_attention_weights')(attention_conv_4)

	attention_representation = multiply([conv4_maxout, attention_weights])
	attention_add = add([conv4_maxout, attention_representation])# 8x6x10
	cnn_flatten = Flatten()(attention_add) #480
	cnn_output = Dense(units=60, activation='tanh', name='cnn_output')(cnn_flatten)# 60


	# classifier module
	ensemble_maxout = maximum([attention_sum, mlp_output, cnn_output])# 60
	emotion_output = Dense(args.classes, activation='softmax', name='emotion_output')(ensemble_maxout)
	
	model = Model(inputs=[dynamic_input, mlp_input, cnn_input], outputs=emotion_output)
	model.summary()

	if args.load_model == '':
		#training
		optimizer = Adam(lr=args.learning_rate, decay=args.decay)
		model.compile(optimizer=optimizer, loss=weighted_loss(r=data_weights), metrics=['accuracy'])
		earlystopping = EarlyStopping(monitor='loss', min_delta=0, patience=0)
		checkpoint = ModelCheckpoint(filepath='dynamic_checkpoint/best_model.hdf5', monitor='loss', save_best_only=True)
		callbacks_list = [earlystopping, checkpoint]
		model.fit(x=[train_data, mlp_train_data, cnn_train_data], y=train_label, batch_size=args.batch_size, epochs=args.train_epochs
							, verbose=2, callbacks=callbacks_list) 
		# save the model
		model.save_weights(args.model_root_dir + 'model.h5')

	else:
		print ('Load the model: {}'.format(args.load_model))
		model.load_weights(args.model_root_dir + args.load_model)
		print('==========================================================')

	#predict
	predict = model.predict([test_data, mlp_test_data, cnn_test_data])
	show_confusion_matrix(predict, test_label, test_size)


if __name__ == '__main__':
	args = get_arguments()
	emotion = ['Anger', 'Emphatic', 'Neutral', 'Positive', 'Rest']
	print('==========================================================')
	
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

	train(train_data, train_label, train_size, test_data, test_label, test_size, max_length)


