
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
import cPickle
from sklearn import svm, datasets
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from keras.utils import np_utils
import os
import scipy as sc
from scipy.io.wavfile import read
import scipy.signal

load_path='../dataset/test/angry' 
file_list = os.listdir(load_path)
d="dataset/train/angry/"
i=1
dodo12=[]	
#dodo12.append([])
for item in  file_list:
    if item.endswith(".wav"):
       print item,i
       i=i+1
       n = os.path.join(load_path, item)
       input = read(n)
       arr = np.array(input[1], dtype=float)
       nparr=sc.signal.resample(arr,1024)
       #arrp=sc.signal.resample(nparr,32)
       dat=nparr.astype(int)
       np.set_printoptions(threshold=np.nan)
       print dat.shape
       data = np.array(dat)
       dodo12.append(data)
X=dodo12
out_file = open("./pkl/dataTestAngry.pkl", 'w+')
dic = {'data':X}
cPickle.dump(dic, out_file)
out_file.close()

