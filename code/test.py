from __future__ import print_function
import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
import cPickle
import scipy as sc
from scipy.io.wavfile import read
import scipy.signal
import os
import numpy as np
from scipy.io.wavfile import read
from keras.models import model_from_json


def save_model(model):
   model_json = model.to_json()
   open('model/moodrec.json', 'w').write(model_json)
   model.save_weights('weights.HDF5', overwrite=True)

def load_model(model_def_fname, model_weight_fname):
   model = model_from_json(open(model_def_fname).read())
   model.load_weights(model_weight_fname)
   return model


def predict(path):
	model=load_model("model/moodRec.json","weight/weights.h5")
	n = os.path.join(path)
        print(path)
	input = read(n)
	arr = np.array(input[1], dtype=float)
	nparr=sc.signal.resample(arr,1024).reshape(1,1,32,32)
	dat=nparr.astype(int)
	np.set_printoptions(threshold=np.nan)
	#print(dat)
	#print dat.shape
	data = np.array(dat)

	prediction=model.predict(data)
	print("Direct Prediction ")
	print(prediction)
	if prediction[0][0]==1:
	   print("Given voice is ANGRY")
	elif prediction[0][1]==1:
	   print("Given voice is NOT ANGRY")
	elif prediction[0][0] > prediction[0][1]:
	   print("Given voice is ANGRY")
	else:
	   print("Given voice is NOT ANGRY")

	class_predict=model.predict_classes(data)
	print("Class Prediction ")
	print(class_predict)

	probability_prediction=model.predict_proba(data)
	print("Probability Prediction ")
	print(probability_prediction)

if __name__=='__main__':
	result=predict("test output/a05.wav")
