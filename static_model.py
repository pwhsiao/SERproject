import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import argparse
import math

from keras.layers import *
from keras.models import *
from keras.optimizers import SGD, Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.merge import *
from utils import *
from keras import backend as K
from keras.models import load_model
from keras.utils import plot_model, print_summary

MODEL_ROOT_DIR = './static_model/'

# Learning parameters
BATCH_SIZE = 100
TRAIN_EPOCHS = 10
LEARNING_RATE = 0.002
DECAY = 0.00001
MOMENTUM = 0.9
# Network parameters
CLASSES = 5 # A,E,N,P,R
LLDS = 16
FUNCTIONALS = 12
DELTA = 2
STATIC_FEATURES = LLDS*FUNCTIONALS*DELTA
HIDDEN_UNITS = 30
USE_TEACHER_LABEL = 0

def get_arguments():
	parser = argparse.ArgumentParser(description='Fau-Aibo with static model.')
	parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
						help='How many train data to process at once. Default: ' + str(BATCH_SIZE) + '.')
	parser.add_argument('--model_root_dir', type=str, default=MODEL_ROOT_DIR,
						help='Root directory to place the trained model.')
	parser.add_argument('--load_model', type=str, default='',
						help='Whether to load existing model for test.')
	parser.add_argument('--use_teacher_label', type=int, default=USE_TEACHER_LABEL,
						help='Whether to use the teacher label.')
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
	parser.add_argument('--static_features', type=int,default=STATIC_FEATURES, 
						help='dimension of features of the static model.')
	parser.add_argument('--hidden_units', type=int, default=HIDDEN_UNITS,
						help='The units of the hidden layer.')
	parser.add_argument('--llds', type=int, default=LLDS,
						help='The llds of the feature.')
	parser.add_argument('--functionals', type=int, default=FUNCTIONALS,
						help='The functionals of the feature.')
	parser.add_argument('--delta', type=int, default=DELTA,
						help='The delta of the feature.')
	return parser.parse_args()


def train_CNN(train_data, train_label, train_size, test_data, test_label, test_size):
		args = get_arguments()
		data_weights = []
		data_weights.append([1.1,0.5,0.2,1.5,1.4])
		data_weights = np.asarray(data_weights)
		class_weight = {0: 1.1, 1: 0.5, 2: 0.2, 3: 1.5, 4: 1.4}

		cnn_train_data = np.reshape(train_data, (train_data.shape[0], args.llds, args.functionals, args.delta))
		cnn_test_data = np.reshape(test_data, (test_data.shape[0], args.llds, args.functionals, args.delta))

		# 8x6x40
		conv1_1 = Conv2D(filters=40,kernel_size=(1,1),strides=(2,2),padding='same', activation='relu', name='conv1_1')(cnn_input)
		#conv1_1 = Dropout(rate=0.3)(conv1_1)
		conv1_2 = Conv2D(filters=40,kernel_size=(3,3),strides=(2,2),padding='same', activation='relu', name='conv1_2')(cnn_input)
		#conv1_2 = Dropout(rate=0.3)(conv1_2)
		conv1_3 = Conv2D(filters=40,kernel_size=(5,5),strides=(2,2),padding='same', activation='relu', name='conv1_3')(cnn_input)
		#conv1_3 = Dropout(rate=0.3)(conv1_3)
		conv1_4 = Conv2D(filters=40,kernel_size=(7,7),strides=(2,2),padding='same', activation='relu', name='conv1_4')(cnn_input)
		#conv1_4 = Dropout(rate=0.3)(conv1_4)
		conv1_5 = Conv2D(filters=40,kernel_size=(9,9),strides=(2,2),padding='same', activation='relu', name='conv1_5')(cnn_input)
		#conv1_5 = Dropout(rate=0.3)(conv1_5)
		conv1_maxout = maximum([conv1_1, conv1_2, conv1_3, conv1_4, conv1_5], name='conv_max1')
		#4x3x30
		conv2_1 = Conv2D(filters=30,kernel_size=(1,1),strides=(1,1),padding='same', activation='relu', name='conv2_1')(conv1_maxout)
		#conv2_1 = Dropout(rate=0.3)(conv2_1)
		conv2_2 = Conv2D(filters=30,kernel_size=(3,3),strides=(1,1),padding='same', activation='relu', name='conv2_2')(conv1_maxout)
		#conv2_2 = Dropout(rate=0.3)(conv2_2)
		conv2_3 = Conv2D(filters=30,kernel_size=(5,5),strides=(1,1),padding='same', activation='relu', name='conv2_3')(conv1_maxout)
		#conv2_3 = Dropout(rate=0.3)(conv2_3)
		conv2_4 = Conv2D(filters=30,kernel_size=(7,7),strides=(1,1),padding='same', activation='relu', name='conv2_4')(conv1_maxout)
		#conv2_4 = Dropout(rate=0.3)(conv2_4)
		conv2_maxout = maximum([conv2_1, conv2_2, conv2_3, conv2_4], name='conv_max2')
		#2x2x20
		conv3_1 = Conv2D(filters=20,kernel_size=(1,1),strides=(1,1),padding='same', activation='relu', name='conv3_1')(conv2_maxout)
		#conv3_1 = Dropout(rate=0.3)(conv3_1)
		conv3_2 = Conv2D(filters=20,kernel_size=(3,3),strides=(1,1),padding='same', activation='relu', name='conv3_2')(conv2_maxout)
		#conv3_2 = Dropout(rate=0.3)(conv3_2)
		conv3_3 = Conv2D(filters=20,kernel_size=(5,5),strides=(1,1),padding='same', activation='relu', name='conv3_3')(conv2_maxout)
		#conv3_3 = Dropout(rate=0.3)(conv3_3)
		conv3_maxout = maximum([conv3_1, conv3_2, conv3_3], name='conv_max3')
		#1x1x10
		conv4_1 = Conv2D(filters=10, kernel_size=(1,1),strides=(1,1),padding='same', activation='relu', name='conv4_1')(conv3_maxout)
		#conv4_1 = Dropout(rate=0.3)(conv4_1)
		conv4_2 = Conv2D(filters=10, kernel_size=(3,3),strides=(1,1),padding='same', activation='relu', name='conv4_2')(conv3_maxout)
		#conv4_2 = Dropout(rate=0.3)(conv4_2)
		conv4_maxout = maximum([conv4_1, conv4_2], name='conv_max4')


		# attention module (input: 8x6x40)
		attention_pool_1 = MaxPooling2D(pool_size=(2,2), padding='same', name='att_pool1')(conv1_maxout)# 4x3x40
		attention_conv_1 = Conv2D(filters=30, kernel_size=(2,2), padding='same', activation='relu', use_bias=False, name='att_conv1')(attention_pool_1)# 4x3x30
		attention_pool_2 = MaxPooling2D(pool_size=(2,2), padding='same', name='att_pool2')(attention_conv_1)#2x2x30
		attention_conv_2 = Conv2D(filters=20, kernel_size=(2,2), padding='same', activation='relu', use_bias=False, name='att_conv2')(attention_pool_2)#2x2x20
		attention_interp_1 = Lambda(interpolation, arguments={'size': (4,3)}, name='att_up1')(attention_conv_2)# 4x3x20
		attention_conv_3 = Conv2D(filters=20, kernel_size=(2,2), padding='same', activation='relu', use_bias=False, name='att_conv3')(attention_interp_1)#4x3x20
		attention_interp_2 = Lambda(interpolation, arguments={'size': (8,6)}, name='att_up2')(attention_conv_3)# 8x6x20
		attention_conv_4 = Conv2D(filters=10, kernel_size=(2,2), padding='same', activation='relu', use_bias=False, name='att_conv4')(attention_interp_2)#8x6x1
		attention_weights = Activation('softmax', name='attention_weights')(attention_conv_4)

		attention_representation = multiply([conv4_maxout, attention_weights])
		attention_add = add([conv4_maxout, attention_representation])
		conv_flatten = Flatten()(conv4_maxout)# 480

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
			model.save_weights(args.model_root_dir + 'model.h5')
		else:
			print ('Load the model: {}'.format(args.load_model))
			model.load_weights(args.model_root_dir + args.load_model)

	
		predict = model.predict(cnn_test_data)
		show_confusion_matrix(predict, test_label, test_size)
		
		
def train_MLP(train_data, train_label, train_size, test_data, test_label, test_size):
	args = get_arguments()
	data_weights = []
	data_weights.append([1.1,0.5,0.2,1.5,1.4])
	data_weights = np.asarray(data_weights)
	class_weight = {0: 1.1, 1: 0.5, 2: 0.2, 3: 1.5, 4: 1.4}

	# 16x12x2
	mlp_input = Input(shape=[args.static_features], dtype='float32', name='mlp_input')
	hidden_1 = Dense(units=30, activation='relu', name='hidden1')(mlp_input)
	hidden_2 = Dense(units=30, activation='relu', name='hidden2')(hidden_1)
	hidden_3 = Dense(units=30, activation='relu', name='hidden3')(hidden_2)
	output = Dense(units=5, activation='softmax', name='output')(hidden_3)
	model = Model(inputs=mlp_input, outputs=output)
	model.summary()

	if args.load_model == '':
		adam = Adam(lr=args.learning_rate, decay=args.decay)
		model.compile(loss=weighted_loss(r=data_weights), optimizer=adam, metrics=['accuracy'])
		earlystopping = EarlyStopping(monitor='loss', min_delta=0, patience=0)
		checkpoint = ModelCheckpoint(filepath='static_model/best_checkpoint_mlp.hdf5', monitor='val_loss', save_best_only=True)
		callbacks_list = [earlystopping, checkpoint]
		history = model.fit(x=train_data, y=train_label, batch_size=args.batch_size, epochs=args.train_epochs, verbose=2, 
							  callbacks=callbacks_list)  
		# save the model
		model.save_weights(args.model_root_dir + 'model.h5')
	else:
		print ('Load the model: {}'.format(args.load_model))
		model.load_weights(args.model_root_dir + args.load_model)

	predict = model.predict(test_data, verbose=1)
	show_confusion_matrix(predict, test_label, test_size)

if __name__ == '__main__':
	args = get_arguments()
	train_data, train_label, train_size, train_info = read_2D_data('./data/CS_Ftrain_nor.arff')
	test_data, test_label, test_size, test_info = read_2D_data('./data/CS_Ftest_nor.arff')
	

	print("Train data：{0} , {1}".format(train_data.shape, train_label.shape))
	print("Test data：{0} , {1}".format(test_data.shape, test_label.shape))
	train_MLP(train_data, train_label, train_size, test_data, test_label, test_size)
	#train_CNN(train_data, train_label, train_size, test_data, test_label, test_size)


	
