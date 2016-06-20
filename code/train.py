from __future__ import print_function
import numpy as np
import h5py
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
import cPickle
import scipy as sc
import scipy.signal
import os
import numpy as np
from scipy.io.wavfile import read
from keras.models import model_from_json
from six.moves import range

class moodDetection:

        __X_train="None"
        __Y_train="None"
        __X_test="None"
        __Y_test="None"
        __batch_size=1
	__nb_epochs=50
	__nb_classes = 2
	__nb_epoch = 20

	# input dimensions
	__img_rows, __img_cols = 32, 32
	# number of convolutional filters to use
	__nb_filters = 32
	# size of pooling area for max pooling
	__nb_pool = 2
	# convolution kernel size
	__nb_conv = 3


	def load_data(self,trainData,trainLabels,testData,testLabels):
               train_file_data = open(trainData, 'r')
	       train=cPickle.load(train_file_data)
	       X_train=np.asarray(train['data'])
	       train_file_data.close()
	       train_file_labels = open(trainLabels, 'r')
	       train_labels=cPickle.load(train_file_labels)
	       Y_train=train_labels['labels']
	       train_file_labels.close()
	       y_train=np.asarray(Y_train)
	       test_file_data = open(testData, 'r')
	       test_data=cPickle.load(test_file_data)
	       X_test=np.asarray(test_data['data'])
	       test_file_data.close()
	       test_file_label = open(testLabels, 'r')
	       test_labels=cPickle.load(test_file_label)
	       y_test=np.asarray(test_labels['labels'])
	       test_file_label.close()
	       self.nb_train_samples = 330

	       X_train = X_train.reshape(X_train.shape[0], 1, self.__img_rows, self.__img_cols)
	       X_test = X_test.reshape(X_test.shape[0], 1, self.__img_rows, self.__img_cols)
	       X_train = X_train.astype('float32')
	       X_test = X_test.astype('float32')
	       X_train /= 255
	       X_test /= 255
	       print('X_train shape:', X_train.shape)
	       print(X_train.shape[0], 'train samples')
	       print(X_test.shape[0], 'test samples')
	       print(np.shape(X_train[3]))
		# convert class vectors to binary class matrices
	       Y_train = np_utils.to_categorical(y_train, self.__nb_classes)
	       Y_test = np_utils.to_categorical(y_test, self.__nb_classes)

               self.X_train=X_train
               self.Y_train=Y_train
               self.X_test=X_test
               self.Y_test=Y_test
               return (X_train,Y_train),(X_test,Y_test)
                
	def save_model(self,model):
	   	model_json = model.to_json()
	   	open('model/moodRec.json', 'w').write(model_json)

	def load_model(self,model_def_fname, model_weight_fname):
	   	model = model_from_json(open(model_def_fname).read())
	   	model.load_weights(model_weight_fname)
	   	return model

	def predict_data_conversion(load_path,file_name):
	   	file_read = os.path.join(load_path, file_name)
	   	input_file = read(file_read)
	   	arr = np.array(input_file[1], dtype=float)
	   	nparr=sc.signal.resample(arr,1024).reshape(1,1,32,32)
	   	predict_data=nparr.astype(int)
	   	return predict_data

        def make_network(self):
	        model = Sequential()
                model.add(Convolution2D(self.__nb_filters, self.__nb_conv, self.__nb_conv,
				        border_mode='valid',
				        input_shape=(1, self.__img_rows, self.__img_cols)))
		model.add(Activation('relu'))
		model.add(Convolution2D(self.__nb_filters, self.__nb_conv, self.__nb_conv))
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=(self.__nb_pool, self.__nb_pool)))
		model.add(Dropout(0.25))
		model.add(Flatten())
		model.add(Dense(128))
		model.add(Activation('relu'))
		model.add(Dropout(0.5))
		model.add(Dense(self.__nb_classes))
		model.add(Activation('softmax'))
		model.compile(loss='categorical_crossentropy',
			      optimizer='adadelta',
			      metrics=['accuracy'])
                return model

	def train_model(self,model,X_train,Y_train,X_test,Y_test):
                
		model.compile(loss='categorical_crossentropy',
			      optimizer='adadelta',
			      metrics=['accuracy'])
		model.fit(X_train, Y_train, self.__batch_size, nb_epoch=100,
		          verbose=1, validation_data=(X_test, Y_test))
		model.save_weights('weights/weights.h5',overwrite=True)
		score = model.evaluate(X_test, Y_test, verbose=1)
 		return score


if __name__=="__main__":
	fyp=moodDetection()
        (X_train,Y_train),(X_test,Y_test)=fyp.load_data("./inputFiles/train_data.pkl","./inputFiles/train_label.pkl","./inputFiles/test_data.pkl","./inputFiles/test_label.pkl")
        print(X_train.shape)
        model=fyp.make_network()
        fyp.save_model(model)
        #model=fyp.load_model("model/moodRec.json","weight/weights.h5")
        score=fyp.train_model(model,X_train,Y_train,X_test,Y_test)
        print(score)

