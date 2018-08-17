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
from scipy import stats
from utils import *

# Learning parameters
BATCH_SIZE = 100
TRAIN_EPOCHS = 15
LEARNING_RATE = 0.002
DECAY = 0.00001
MOMENTUM = 0.9
# Network parameters
CLASSES = 5 # A,E,N,P,R
LLDS = 16
FUNCTIONALS = 12
DELTA = 2
STATIC_FEATURES = LLDS*FUNCTIONALS*DELTA
DYNAMIC_FEATURES = 32

MODEL_ROOT_DIR = './bagging_model'
BAGGING_ITERS = 10
BASE_CLASSIFIER = 'CNN'
SAMPLE_SCHEME = 'BALANCED'

def get_arguments():
	parser = argparse.ArgumentParser(description='Fau-Aibo with bagging ensemble model.')
	parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
						help='How many train data to process at once. Default: ' + str(BATCH_SIZE) + '.')
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
						help='number of classes in network.')
	parser.add_argument('--dynamic_features', type=int,default=DYNAMIC_FEATURES, 
						help='dimension of features of the dynamic model.')
	parser.add_argument('--static_features', type=int,default=STATIC_FEATURES, 
						help='dimension of features of the static model.')
	parser.add_argument('--llds', type=int, default=LLDS,
						help='The llds of the feature.')
	parser.add_argument('--functionals', type=int, default=FUNCTIONALS,
						help='The functionals of the feature.')
	parser.add_argument('--delta', type=int, default=DELTA,
						help='The delta of the feature.')
	parser.add_argument('--model_root_dir', type=str, default=MODEL_ROOT_DIR,
						help='Root directory to place the trained model.')
	parser.add_argument('--bagging_iters', type=int, default=BAGGING_ITERS,
						help='Number of the bagging performed.')
	parser.add_argument('--base_classifier', type=str, default=BASE_CLASSIFIER,
						help='the base classifier applied by bagging.')
	parser.add_argument('--sample_scheme', type=str, default=SAMPLE_SCHEME,
						help='whether the sample data is balanced or unbalanced(skewed).')
	return parser.parse_args()

def adjust_data_weights(data, label):
	data_weights = []
	a, e, n, p ,r = separa_fau_five(data, label)
	a_w = round((data.shape[0]/a.shape[0])/10, 1)
	e_w = round((data.shape[0]/e.shape[0])/10, 1)
	n_w = round((data.shape[0]/n.shape[0])/10, 1)
	p_w = round((data.shape[0]/p.shape[0])/10, 1)
	r_w = round((data.shape[0]/r.shape[0])/10, 1)
	data_weights.append([a_w, e_w, n_w, p_w, r_w])
	data_weights = np.asarray(data_weights)
	return data_weights

def balanced_bagging_sample(data, label, emotion_name_list):
	bagging_train_data = []
	bagging_train_label = []
	a, e, n, p ,r = separa_fau_five(data, label)
	bagging_train_size = np.zeros(5)
	is_selected = np.zeros(data.shape[0])
	# A_duplicated_sample = np.zeros(a.shape[0])
	# E_duplicated_sample = np.zeros(e.shape[0])
	# N_duplicated_sample = np.zeros(n.shape[0])
	# P_duplicated_sample = np.zeros(p.shape[0])
	# R_duplicated_sample = np.zeros(r.shape[0])
	for i in range(data.shape[0]):
		emotion_name = np.random.choice(emotion_name_list, p=[0.2,0.2,0.2,0.2,0.2])
		if emotion_name == 'Angry':
			random = np.random.randint(0, a.shape[0])
			bagging_train_data.append(a[random])
			bagging_train_label.append([1,0,0,0,0])
			bagging_train_size[0] += 1
			#A_duplicated_sample[random] += 1
		elif emotion_name == 'Emphatic':
			random = np.random.randint(0, e.shape[0])
			bagging_train_data.append(e[random])
			bagging_train_label.append([0,1,0,0,0])
			bagging_train_size[1] += 1
			#E_duplicated_sample[random] += 1
		elif emotion_name == 'Neutral':
			random = np.random.randint(0, n.shape[0])
			bagging_train_data.append(n[random])
			bagging_train_label.append([0,0,1,0,0])
			bagging_train_size[2] += 1
			#N_duplicated_sample[random] += 1
		elif emotion_name == 'Positive':
			random = np.random.randint(0, p.shape[0])
			bagging_train_data.append(p[random])
			bagging_train_label.append([0,0,0,1,0])
			bagging_train_size[3] += 1
			#P_duplicated_sample[random] += 1
		elif emotion_name == 'Rest':
			random = np.random.randint(0, r.shape[0])
			bagging_train_data.append(r[random])
			bagging_train_label.append([0,0,0,0,1])
			bagging_train_size[4] += 1
			#R_duplicated_sample[random] += 1
	bagging_train_data = np.asarray(bagging_train_data)
	bagging_train_label = np.asarray(bagging_train_label)
	# print (A_duplicated_sample)
	# print (E_duplicated_sample)
	# print (N_duplicated_sample)
	# print (P_duplicated_sample)
	# print (R_duplicated_sample)
	return bagging_train_data, bagging_train_label, bagging_train_size


def skewed_bagging_sample(data, label, emotion_name_list):
	bagging_train_data = []
	bagging_train_label = []
	a, e, n, p ,r = separa_fau_five(data, label)
	bagging_train_size = np.zeros(5)
	for i in range(data.shape[0]):
		random = np.random.randint(0, data.shape[0])
		bagging_train_data.append(data[random])
		bagging_train_label.append(label[random])
		bagging_train_size[np.argmax(label[random])] += 1
		
	bagging_train_data = np.asarray(bagging_train_data)
	bagging_train_label = np.asarray(bagging_train_label)
	
	return bagging_train_data, bagging_train_label, bagging_train_size


def voting(predicts):# (k, 8257, 5)
	duplicated_count = 0
	non_duplicated_count = 0
	final_predict = []
	for i in range(predicts.shape[1]):
		majority_vote = np.zeros(5)
		# each classifier votes on the ith data
		for j in range(predicts.shape[0]):
			if np.argmax(predicts[j,i,:]) == 0:
				majority_vote[0] += 1
			elif np.argmax(predicts[j,i,:]) == 1:
				majority_vote[1] += 1
			elif np.argmax(predicts[j,i,:]) == 2:
				majority_vote[2] += 1
			elif np.argmax(predicts[j,i,:]) == 3:
				majority_vote[3] += 1
			elif np.argmax(predicts[j,i,:]) == 4:
				majority_vote[4] += 1
		# summarize the voting results
		winner = np.argwhere(majority_vote == np.amax(majority_vote))
		# the highest votes are not duplicated
		if winner.shape[0] == 1:
			non_duplicated_count += 1
			if np.argmax(majority_vote) == 0:
				final_predict.append([1,0,0,0,0])
			elif np.argmax(majority_vote) == 1:
				final_predict.append([0,1,0,0,0])
			elif np.argmax(majority_vote) == 2:
				final_predict.append([0,0,1,0,0])
			elif np.argmax(majority_vote) == 3:
				final_predict.append([0,0,0,1,0])
			elif np.argmax(majority_vote) == 4:
				final_predict.append([0,0,0,0,1])
		# the highest number of votes is duplicated, choose randomly or average the predictions
		else:
			#avg_predict = np.mean(predicts[0:predicts.shape[0], i], axis=0)
			#avg_predict = avg_predict.tolist()
			duplicated_count += 1
			random_choice = np.random.choice(winner.flatten())
			temp = [0,0,0,0,0]
			temp[random_choice] += 1
			final_predict.append(temp)
	final_predict = np.asarray(final_predict)
	print ('Number of votes duplicated: {} , number of votes not duplicated: {}'.format(duplicated_count, non_duplicated_count))
	return final_predict

def MCAR_missing(data, label, ratio):
	missing_index = []
	for idx in range(data.shape[0]):
		observed = np.random.binomial(n=1, p=1-ratio)
		if observed == 1:
			pass
		else:
			missing_index.append(idx)
	data = np.delete(data, missing_index, axis=0)
	label = np.delete(label, missing_index, axis=0)
	return data, label

def train_RNN(train_data, train_label, test_data, test_label, test_size, max_length, data_weights, bagging_model_path=''):
	# use LSTM model for dynamic model
	args = get_arguments()

	dynamic_input = Input(shape=[max_length, args.dynamic_features], dtype='float32', name='dynamic_input')
	# LSTM module
	lstm1 = LSTM(units=60, activation='tanh', return_sequences=True, recurrent_dropout=0.5, name='lstm1')(dynamic_input_dropout)	
	lstm1 = Dropout(rate=0.5, noise_shape=(1, 60))(lstm1)
	
	# attention mechanism module 
	attention_dense1_f = Dense(units=50, activation='tanh', use_bias=False, name='attention_dense1_f')(lstm1)
	attention_dense2_f = Dense(1, use_bias=False, name='attention_dense2_f')(attention_dense1_f)
	attention_flatten_f = Flatten()(attention_dense2_f)
	attention_softmax_f = Activation('softmax', name='attention_weights_f')(attention_flatten_f)
	attention_repeat_f = RepeatVector(60)(attention_softmax_f)
	attention_permute_f = Permute([2, 1])(attention_repeat_f)
	attention_multiply_f = multiply([lstm1, attention_permute_f])
	attention_sum_f = Lambda(lambda xin: K.sum(xin, axis=1), name='attention_sum_f')(attention_multiply_f)# 60

	
	
	# classifier module
	#output = average([output_f, output_b])
	output = Dense(args.classes, activation='softmax', name='output')(attention_sum_f)
	model = Model(inputs=dynamic_input, outputs=output)
	model.summary()

	if args.load_model == '':
		#training
		optimizer = Adam(lr=args.learning_rate, decay=args.decay)
		#optimizer = RMSprop()
		model.compile(loss=weighted_loss(r=data_weights), optimizer=optimizer, metrics=['accuracy'])
		#model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
		earlystopping = EarlyStopping(monitor='loss', min_delta=0, patience=0)
		checkpoint = ModelCheckpoint(filepath='dynamic_model/best_checkpoint.hdf5', monitor='val_loss', save_best_only=True)
		callbacks_list = [earlystopping]
		model.fit(x=train_data, y=train_label, batch_size=args.batch_size, epochs=args.train_epochs
							, verbose=2, callbacks=callbacks_list) 
		# save the model
		if bagging_model_path != '':
			bagging_model_path = bagging_model_path + '.h5'
			model.save_weights(bagging_model_path)
	else:
		print ('Load the model: {}'.format(args.load_model))
		model.load_weights(args.model_root_dir + args.load_model)
		print('==========================================================')
	
	#predict
	predict = model.predict(test_data)
	return predict
	


def train_CNN(train_data, train_label, test_data, test_label, test_size, data_weights, bagging_model_path=''):
		args = get_arguments()

		cnn_train_data = np.reshape(train_data, (train_data.shape[0], args.llds, args.functionals, args.delta))
		cnn_test_data = np.reshape(test_data, (test_data.shape[0], args.llds, args.functionals, args.delta))

		# 16x12x2
		cnn_input = Input(shape=[args.llds, args.functionals, args.delta], dtype='float32', name='cnn_input')
		#cnn_input_dropout = Dropout(rate=0.2)(cnn_input)
		# 8x6x40
		conv1_1 = Conv2D(filters=40,kernel_size=(1,1),strides=(2,2),padding='same', activation='tanh', name='conv1_1')(cnn_input)
		#conv1_1 = Dropout(rate=0.5)(conv1_1)
		conv1_2 = Conv2D(filters=40,kernel_size=(3,3),strides=(2,2),padding='same', activation='tanh', name='conv1_2')(cnn_input)
		#conv1_2 = Dropout(rate=0.5)(conv1_2)
		conv1_3 = Conv2D(filters=40,kernel_size=(5,5),strides=(2,2),padding='same', activation='tanh', name='conv1_3')(cnn_input)
		#conv1_3 = Dropout(rate=0.5)(conv1_3)
		conv1_4 = Conv2D(filters=40,kernel_size=(7,7),strides=(2,2),padding='same', activation='tanh', name='conv1_4')(cnn_input)
		#conv1_4 = Dropout(rate=0.5)(conv1_4)
		conv1_5 = Conv2D(filters=40,kernel_size=(9,9),strides=(2,2),padding='same', activation='tanh', name='conv1_5')(cnn_input)
		#conv1_5 = Dropout(rate=0.5)(conv1_5)
		conv1_maxout = maximum([conv1_1, conv1_2, conv1_3, conv1_4, conv1_5], name='conv_max1')
		conv1_maxout = Dropout(rate=0.5)(conv1_maxout)
		#4x3x30
		conv2_1 = Conv2D(filters=30,kernel_size=(1,1),strides=(1,1),padding='same', activation='tanh', name='conv2_1')(conv1_maxout)
		#conv2_1 = Dropout(rate=0.5)(conv2_1)
		conv2_2 = Conv2D(filters=30,kernel_size=(3,3),strides=(1,1),padding='same', activation='tanh', name='conv2_2')(conv1_maxout)
		#conv2_2 = Dropout(rate=0.5)(conv2_2)
		conv2_3 = Conv2D(filters=30,kernel_size=(5,5),strides=(1,1),padding='same', activation='tanh', name='conv2_3')(conv1_maxout)
		#conv2_3 = Dropout(rate=0.5)(conv2_3)
		conv2_4 = Conv2D(filters=30,kernel_size=(7,7),strides=(1,1),padding='same', activation='tanh', name='conv2_4')(conv1_maxout)
		#conv2_4 = Dropout(rate=0.5)(conv2_4)
		conv2_maxout = maximum([conv2_1, conv2_2, conv2_3, conv2_4], name='conv_max2')
		conv2_maxout = Dropout(rate=0.5)(conv2_maxout)
		#2x2x20
		conv3_1 = Conv2D(filters=20,kernel_size=(1,1),strides=(1,1),padding='same', activation='tanh', name='conv3_1')(conv2_maxout)
		#conv3_1 = Dropout(rate=0.5)(conv3_1)
		conv3_2 = Conv2D(filters=20,kernel_size=(3,3),strides=(1,1),padding='same', activation='tanh', name='conv3_2')(conv2_maxout)
		#conv3_2 = Dropout(rate=0.5)(conv3_2)
		conv3_3 = Conv2D(filters=20,kernel_size=(5,5),strides=(1,1),padding='same', activation='tanh', name='conv3_3')(conv2_maxout)
		#conv3_3 = Dropout(rate=0.5)(conv3_3)
		conv3_maxout = maximum([conv3_1, conv3_2, conv3_3], name='conv_max3')
		conv3_maxout = Dropout(rate=0.5)(conv3_maxout)
		#1x1x10
		conv4_1 = Conv2D(filters=10, kernel_size=(1,1),strides=(1,1),padding='same', activation='tanh', name='conv4_1')(conv3_maxout)
		#conv4_1 = Dropout(rate=0.5)(conv4_1)
		conv4_2 = Conv2D(filters=10, kernel_size=(3,3),strides=(1,1),padding='same', activation='tanh', name='conv4_2')(conv3_maxout)
		#conv4_2 = Dropout(rate=0.5)(conv4_2)
		conv4_maxout = maximum([conv4_1, conv4_2], name='conv_max4')
		conv4_maxout = Dropout(rate=0.5)(conv4_maxout)


		# attention module (input: 8x6x40)
		attention_pool_1 = MaxPooling2D(pool_size=(2,2), padding='same', name='att_pool1')(conv1_maxout)# 4x3x40
		attention_conv_1 = Conv2D(filters=30, kernel_size=(2,2), padding='same', activation='tanh', use_bias=False, name='att_conv1')(attention_pool_1)# 4x3x30
		#attention_conv_1 = Dropout(rate=0.5)(attention_conv_1)
		attention_pool_2 = MaxPooling2D(pool_size=(2,2), padding='same', name='att_pool2')(attention_conv_1)#2x2x30
		attention_conv_2 = Conv2D(filters=20, kernel_size=(2,2), padding='same', activation='tanh', use_bias=False, name='att_conv2')(attention_pool_2)#2x2x20
		attention_conv_2 = Dropout(rate=0.5)(attention_conv_2)
		attention_interp_1 = Lambda(interpolation, arguments={'size': (4,3)}, name='att_up1')(attention_conv_2)# 4x3x20
		attention_conv_3 = Conv2D(filters=20, kernel_size=(2,2), padding='same', activation='tanh', use_bias=False, name='att_conv3')(attention_interp_1)#4x3x20
		attention_conv_3 = Dropout(rate=0.5)(attention_conv_3)
		attention_interp_2 = Lambda(interpolation, arguments={'size': (8,6)}, name='att_up2')(attention_conv_3)# 8x6x20
		attention_conv_4 = Conv2D(filters=10, kernel_size=(2,2), padding='same', activation='tanh', use_bias=False, name='att_conv4')(attention_interp_2)#8x6x1
		attention_conv_4 = Dropout(rate=0.5)(attention_conv_4)
		attention_weights = Activation('softmax', name='attention_weights')(attention_conv_4)
		

		attention_representation = multiply([conv4_maxout, attention_weights])
		attention_add = add([conv4_maxout, attention_representation])
		conv_flatten = Flatten()(attention_add)# 480

		output = Dense(units=args.classes, activation='softmax', name='output')(conv_flatten)
		model = Model(inputs=cnn_input, outputs=output)
		model.summary()

		if args.load_model == '':
			adam = Adam(lr=args.learning_rate, decay=args.decay)
			model.compile(loss=weighted_loss(r=data_weights), optimizer=adam, metrics=['accuracy'])
			#model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
			earlystopping = EarlyStopping(monitor='loss', min_delta=0, patience=0)
			callbacks_list = [earlystopping]
			history = model.fit(x=cnn_train_data, y=train_label, batch_size=args.batch_size, epochs=args.train_epochs, verbose=2, 
						          callbacks=callbacks_list)  
			# save the model
			if bagging_model_path != '':
				bagging_model_path = bagging_model_path + '.h5'
				model.save_weights(bagging_model_path)
		else:
			print ('Load the model: {}'.format(args.load_model))
			model.load_weights(args.model_root_dir + args.load_model)

		predict = model.predict(cnn_test_data)
		return predict
		
def train_MLP(train_data, train_label, test_data, test_label, test_size, data_weights, bagging_model_path=''):
	args = get_arguments()
	

	# 16x12x2
	mlp_input = Input(shape=[args.static_features], dtype='float32', name='mlp_input')
	#mlp_input_dr = Dropout(rate=0.2)(mlp_input)
	hidden_1 = DropConnect(Dense(units=30, activation='tanh', name='hidden_1'), prob=0.4)(mlp_input)
	#hidden_1 = Dropout(rate=0.5)(hidden_1)
	output = Dense(units=5, activation='softmax', name='output')(hidden_1)
	model = Model(inputs=mlp_input, outputs=output)
	model.summary()

	if args.load_model == '':
		adam = Adam(lr=args.learning_rate, decay=args.decay)
		model.compile(loss=weighted_loss(r=data_weights), optimizer=adam, metrics=['accuracy'])
		earlystopping = EarlyStopping(monitor='loss', min_delta=0, patience=0)
		callbacks_list = [earlystopping]
		history = model.fit(x=train_data, y=train_label, batch_size=args.batch_size, epochs=args.train_epochs, verbose=2, 
					          callbacks=callbacks_list)  
		# save the model
		if bagging_model_path != '':
			bagging_model_path = bagging_model_path + '.h5'
			model.save_weights(bagging_model_path)
	else:
		print ('Load the model: {}'.format(args.load_model))
		model.load_weights(args.model_root_dir + args.load_model)
	
	predict = model.predict(test_data)
	#show_confusion_matrix(predict, test_label, test_size)
	return predict


if __name__ == '__main__':
	args = get_arguments()
	emotion_name_list = ['Angry', 'Emphatic', 'Neutral', 'Positive', 'Rest']
	print('==========================================================')
	
	dynamic_train_data = np.load('./data/fau_train_lld_delta_fea_sn_pad.npy')
	dynamic_test_data = np.load('./data/fau_test_lld_delta_fea_sn_pad.npy')
	
	static_train_data, train_label, train_size, train_info = read_2D_data('./data/CS_Ftrain_nor.arff')
	static_test_data, test_label, test_size, test_info = read_2D_data('./data/CS_Ftest_nor.arff')

	train_seq_length = np.load('./data/fau_train_length_lld_delta.npy')
	test_seq_length = np.load('./data/fau_test_length_lld_delta.npy')
	max_length = max(np.amax(train_seq_length), np.amax(test_seq_length))


	print('==========================================================')
	print ('Dynamic:')
	print('Original train data and label: {} , {}'.format(dynamic_train_data.shape, train_label.shape))
	print('Test data and label: {} , {}'.format(dynamic_test_data.shape, test_label.shape))
	print('==========================================================')
	print ('Static:')
	print('Original train data and label: {} , {}'.format(static_train_data.shape, train_label.shape))
	print('Test data and label: {} , {}'.format(static_test_data.shape, test_label.shape))
	print('==========================================================')

	predicts = np.zeros((args.bagging_iters, dynamic_test_data.shape[0], 5))# (k, 8257, 5)
	data_weights = [np.ones(5)]
	data_weights = np.asarray(data_weights)
	if args.base_classifier.upper() == 'RNN':
		print ('<< Bagging training procedure for RNN >>')
		args.model_root_dir = os.path.join(args.model_root_dir, 'RNN')
		temp = args.model_root_dir
		for i in range(args.bagging_iters):
			model_num = 'model_' + str(i+1)
			args.model_root_dir = os.path.join(args.model_root_dir, model_num)
			print ('\nConstruct dataset for classifier {}/{}........'.format(i+1, args.bagging_iters))
			print('==========================================================')
			if args.sample_scheme[0] == 'b':
				bag_dynamic_train_data, bag_train_label, bag_train_size = balanced_bagging_sample(dynamic_train_data, train_label, emotion_name_list)
			elif args.sample_scheme[0] == 'u':
				bag_dynamic_train_data, bag_train_label, bag_train_size = skewed_bagging_sample(dynamic_train_data, train_label, emotion_name_list)
				data_weights = adjust_data_weights(bag_dynamic_train_data, bag_train_label)
			print ('Data sizes for each class: {}'.format(bag_train_size))
			bag_dynamic_train_data, bag_train_label = MCAR_missing(bag_dynamic_train_data, bag_train_label, 0.05)
			print ('After MCAR missing...')
			print (bag_dynamic_train_data.shape , bag_train_label.shape)
			predict = train_RNN(bag_dynamic_train_data, bag_train_label, dynamic_test_data, test_label, test_size, max_length, data_weights, args.model_root_dir)
			predicts[i] = predict
			args.model_root_dir = temp

	elif args.base_classifier.upper() == 'CNN':
		print ('<< Bagging training procedure for CNN >>')
		args.model_root_dir = os.path.join(args.model_root_dir, 'CNN')
		temp = args.model_root_dir
		for i in range(args.bagging_iters):
			model_num = 'model_' + str(i+1)
			args.model_root_dir = os.path.join(args.model_root_dir, model_num)
			print ('\nConstruct dataset for classifier {}/{}........'.format(i+1, args.bagging_iters))
			print('==========================================================')
			if args.sample_scheme[0] == 'b':
				bag_static_train_data, bag_train_label, bag_train_size = balanced_bagging_sample(static_train_data, train_label, emotion_name_list)
			elif args.sample_scheme[0] == 'u':
				bag_static_train_data, bag_train_label, bag_train_size = skewed_bagging_sample(static_train_data, train_label, emotion_name_list)
				data_weights = adjust_data_weights(bag_static_train_data, bag_train_label)
			print ('Data sizes for each class: {}'.format(bag_train_size))
			bag_static_train_data, bag_train_label = MCAR_missing(bag_static_train_data, bag_train_label, 0.05)
			print ('After MCAR missing...')
			print (bag_static_train_data.shape , bag_train_label.shape)
			predict = train_CNN(bag_static_train_data, bag_train_label, static_test_data, test_label, test_size, data_weights, args.model_root_dir)
			predicts[i] = predict
			args.model_root_dir = temp

	elif args.base_classifier.upper() == 'MLP':
		print ('<< Bagging training procedure for MLP >>')
		args.model_root_dir = os.path.join(args.model_root_dir, 'MLP')
		temp = args.model_root_dir
		for i in range(args.bagging_iters):
			model_num = 'model_' + str(i+1)
			drop_rate = np.random.random_sample()
			args.model_root_dir = os.path.join(args.model_root_dir, model_num)
			print ('\nConstruct dataset for classifier {}/{}........'.format(i+1, args.bagging_iters))
			if args.sample_scheme[0] == 'b':# balanced sample scheme
				print ('Balanced sample scheme')
				bag_static_train_data, bag_train_label, bag_train_size = balanced_bagging_sample(static_train_data, train_label, emotion_name_list)
			elif args.sample_scheme[0] == 'u':# unbalanced sample scheme
				print ('Unbalanced sample scheme')
				bag_static_train_data, bag_train_label, bag_train_size = skewed_bagging_sample(static_train_data, train_label, emotion_name_list)
				data_weights = adjust_data_weights(bag_static_train_data, bag_train_label)
			print ('Data sizes for each class: {}'.format(bag_train_size))
			bag_static_train_data, bag_train_label = MCAR_missing(bag_static_train_data, bag_train_label, 0.05)
			print ('After MCAR missing...')
			print (bag_static_train_data.shape , bag_train_label.shape)
			predict = train_MLP(bag_static_train_data, bag_train_label, static_test_data, test_label, test_size, data_weights, args.model_root_dir)
			predicts[i] = predict
			args.model_root_dir = temp
			print('==========================================================')

	final_predict = voting(predicts)
	show_confusion_matrix(final_predict, test_label, test_size)
	
