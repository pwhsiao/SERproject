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
from nested_lstm import NestedLSTM



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


def gaussian_attention(inputs, batch_size):
	lower, upper = 0.0, 1.0
	mu, sigma = 0.0, 0.05
	temp = np.zeros(2453, dtype=np.float32)
	normal_attention = K.zeros((inputs.get_shape()[0], 2453))
	lbound, rbound = -3, 3
	gap = (rbound-lbound)/2453
	x = lbound
	i = 0
	while x <= rbound:
		temp[i] = (1.0/(np.sqrt(2.0*np.pi*np.square(sigma)))) * np.exp(-(np.square(x-mu))/(2.0*np.square(sigma)))
		i += 1
		x += gap

	_max = np.max(temp)
	_min = np.min(temp)
	for i in range(2453):
		temp[i] = (((temp[i] - _min) * (upper - lower)) / (_max - _min)) + lower

	_sum = np.sum(temp)
	for i in range(2453):
		temp[i] = temp[i] / _sum
	print (inputs.get_shape()[0])
	temp = np.repeat(temp[np.newaxis,:], inputs.get_shape()[0], axis=0)
	K.set_value(normal_attention, temp)
	#print (K.is_keras_tensor(normal_attention))
	return normal_attention

def train_RNN(train_data, train_label, train_size,test_data, test_label, test_size, max_length, bagging_model_path=''):
	# use LSTM model for dynamic model
	args = get_arguments()
	class_weight = {0: 1.1, 1: 0.5, 2: 0.2, 3: 1.5, 4: 1.4}
	data_weights = []
	data_weights.append([1.1, 0.5, 0.2, 1.5, 1.4])# original
	data_weights = np.asarray(data_weights)

	dynamic_input = Input(shape=[max_length, args.dynamic_features], dtype='float32', name='dynamic_input')

	lstm1 = LSTM(60, activation='tanh', return_sequences=True, recurrent_dropout=0.5, name='lstm1')(dynamic_input)
	#lstm2 = LSTM(32, activation='tanh', return_sequences=True, recurrent_dropout=0.5, name='lstm2')(lstm1)
	#blstm_f, blstm_b = Bidirectional(LSTM(units=args.lstm_units, activation='tanh', return_sequences=True, recurrent_dropout=0.5), merge_mode=None ,name='blstm_1')(dynamic_input_dropout)
	#blstm_out = maximum([blstm_f, blstm_b])
	
	#attention mechanism module (forward)
	# attention_dense1_f = Dense(args.attention_layer_units, activation='tanh', use_bias=False, name='attention_dense1_f')(lstm1)
	# attention_dense2_f = Dense(1, use_bias=False, name='attention_dense2_f')(attention_dense1_f)
	# attention_flatten_f = Flatten()(attention_dense2_f)
	# attention_softmax_f = Activation('softmax', name='attention_weights_f')(attention_flatten_f)
	attention_softmax_f = Lambda(gaussian_attention, arguments={'batch_size': args.batch_size})(lstm1)
	attention_repeat_f = RepeatVector(args.lstm_units)(attention_softmax_f)
	attention_permute_f = Permute([2, 1])(attention_repeat_f)

	attention_multiply_f = multiply([lstm1, attention_permute_f])
	attention_representation = Lambda(lambda xin: K.sum(xin, axis=1), name='attention_representation')(attention_multiply_f)# 60

	'''
	# attention mechanism module (backward)
	attention_dense1_b = Dense(args.attention_layer_units, activation='tanh', use_bias=False, name='attention_dense1_b')(blstm_b)
	attention_dense2_b = Dense(1, use_bias=False, name='attention_dense2_b')(attention_dense1_b)
	attention_flatten_b = Flatten()(attention_dense2_b)
	attention_softmax_b = Activation('softmax', name='attention_weights_b')(attention_flatten_b)
	attention_repeat_b = RepeatVector(args.lstm_units)(attention_softmax_b)
	attention_permute_b = Permute([2, 1])(attention_repeat_b)
	attention_multiply_b = multiply([blstm_b, attention_permute_b])
	attention_sum_b = Lambda(lambda xin: K.sum(xin, axis=1), name='attention_sum_b')(attention_multiply_b)# 60
	
	
	attention_weights_avg = average([attention_softmax_f, attention_softmax_b])
	attention_repeat = RepeatVector(args.lstm_units)(attention_weights_avg)
	attention_permute = Permute([2, 1])(attention_repeat)
	attention_multiply = multiply([blstm_out, attention_permute])
	attention_sum = Lambda(lambda xin: K.sum(xin, axis=1), name='attention_sum')(attention_multiply)# 60
	'''
	#fb_merge = concatenate([attention_sum_f, attention_sum_b])
	
	# classifier module
	#mean = Lambda(lambda xin: K.mean(xin, axis=1))(lstm2)
	output = Dense(args.classes, activation='softmax', name='output')(attention_representation)
	model = Model(inputs=dynamic_input, outputs=output)
	model.summary()

	#opts1 = tf.profiler.ProfileOptionBuilder.float_operation()
	#flops = tf.profiler.profile(tf.get_default_graph(), run_meta=tf.RunMetadata(), cmd='op', options=opts1)
	#print('Flops: {}'.format(flops.total_float_ops))
	#tensorboard = TensorBoard(log_dir='./dynamic_model_log', histogram_freq=0, write_graph=True, write_images=True)
	if args.load_model == '':
		#training
		optimizer = Adam(lr=args.learning_rate, decay=args.decay)
		#model.compile(loss=weighted_loss(r=data_weights), optimizer=optimizer, metrics=['accuracy'])
		model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
		earlystopping = EarlyStopping(monitor='loss', min_delta=0, patience=0)
		checkpoint = ModelCheckpoint(filepath='dynamic_model/best_checkpoint.hdf5', monitor='val_loss', save_best_only=True)
		callbacks_list = [earlystopping]
		model.fit(x=train_data, y=train_label, batch_size=args.batch_size, epochs=args.train_epochs
							, verbose=2, class_weight=class_weight) 
		# save the model
		model.save_weights(args.model_root_dir + 'model.h5')
	else:
		print ('Load the model: {}'.format(args.load_model))
		model.load_weights(args.model_root_dir + args.load_model)
		print('==========================================================')
	'''
	get_attention_weights_f = K.function([model.get_layer(name='blstm_1').input, K.learning_phase()], [model.get_layer(name='attention_weights_f').output])
	attention_weights_f = []
	i = 0
	while i < test_data.shape[0]: 
		if i + args.batch_size >= test_data.shape[0]:
			temp = get_attention_weights_f([test_data[i:test_data.shape[0]], 0])[0]
		temp = get_attention_weights_f([test_data[i:i+args.batch_size], 0])[0]
		temp = np.asarray(temp)
		i += args.batch_size
		for j in range(temp.shape[0]):
			attention_weights_f.append(temp[j])
	attention_weights_f = np.asarray(attention_weights_f)
	print (attention_weights_f.shape)
	np.save('attention_weights_f.npy', attention_weights_f)

	get_attention_weights_b = K.function([model.get_layer(name='blstm_1').input, K.learning_phase()], [model.get_layer(name='attention_weights_b').output])
	attention_weights_b = []
	i = 0
	while i < test_data.shape[0]: 
		if i + args.batch_size >= test_data.shape[0]:
			temp = get_attention_weights_b([test_data[i:test_data.shape[0]], 0])[0]
		temp = get_attention_weights_b([test_data[i:i+args.batch_size], 0])[0]
		temp = np.asarray(temp)
		i += args.batch_size
		for j in range(temp.shape[0]):
			attention_weights_b.append(temp[j])
	attention_weights_b = np.asarray(attention_weights_b)
	print (attention_weights_b.shape)
	np.save('attention_weights_b.npy', attention_weights_b)
	'''
	#predict
	predict = model.predict(test_data)
	show_confusion_matrix(predict, test_label, test_size)

if __name__ == '__main__':
	args = get_arguments()
	emotion_name_list = ['Angry', 'Emphatic', 'Neutral', 'Positive', 'Rest']
	print('==========================================================')
	# No normalize
	
	if args.normalize_mode == 0:
		train_data = np.load('./data/fau_train_lld_delta_pad.npy')
		test_data = np.load('./data/fau_test_lld_delta_pad.npy')
		print ('Normalization mode: none')
	# Feature-wise normalize
	elif args.normalize_mode == 1:
		train_data = np.load('./data/fau_train_lld_delta_feanor_-1_1_pad.npy')
		test_data = np.load('./data/fau_test_lld_delta_feanor_-1_1_pad.npy')
		print ('Normalization mode: feature-wise ')
	# Audio-wise normalize
	elif args.normalize_mode == 2:
		train_data = np.load('./data/fau_train_lld_delta_audnor_-1_1_pad.npy')
		test_data = np.load('./data/fau_test_lld_delta_audnor_-1_1_pad.npy')
		print ('Normalization mode: audio-wise')
	elif args.normalize_mode == 3:
		train_data = np.load('./data/fau_train_lld_delta_fea_sn_pad.npy')
		test_data = np.load('./data/fau_test_lld_delta_fea_sn_pad.npy')
		#train_data = np.load('modified_dynamic_fau_train_cmvn_sn_pad.npy')
		#test_data = np.load('modified_dynamic_fau_test_cmvn_sn_pad.npy')
		print ('Normalization mode: speaker-wise')

	#train_label = np.load('./data/fau_train_label.npy')
	#train_label = np.load('bi_att_teacher_label.npy')
	train_label = np.load('dynamic_teacher_label_4634.npy')
	train_size = np.array([881, 2093, 5590, 674, 721])
	train_seq_length = np.load('./data/fau_train_length_lld_delta.npy')
	test_label = np.load('./data/fau_test_label.npy')
	test_size = np.array([611, 1508, 5377, 215, 546])
	test_seq_length = np.load('./data/fau_test_length_lld_delta.npy')
	max_length = max(np.amax(train_seq_length), np.amax(test_seq_length))

	#train_data = sequence.pad_sequences(train_data, maxlen=max_length, dtype='float32')
	#test_data = sequence.pad_sequences(test_data, maxlen=max_length, dtype='float32')

	
	print('==========================================================')
	print('Train data and label: {} , {}'.format(train_data.shape, train_label.shape))
	print('Test data and label: {} , {}'.format(test_data.shape, test_label.shape))
	print('==========================================================')

	
	train_RNN(train_data, train_label, train_size, test_data, test_label, test_size, max_length)


