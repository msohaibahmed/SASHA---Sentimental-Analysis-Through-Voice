from keras.utils.visualize_util import plot
from keras.models import Sequential
from keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D
import numpy as np
from scipy.misc import imsave
from keras import backend as K
from scipy.io.wavfile import read
from keras.models import model_from_json
import h5py

def load_model(model_def_fname, model_weight_fname):
	   	model = model_from_json(open(model_def_fname).read())
	   	model.load_weights(model_weight_fname)
	   	return model

model=load_model("moodRec.json","weights.HDF5")
plot(model, to_file='model.png')
