import os
import json
import cPickle
import numpy as np
from PIL import Image
from PIL import ImageColor
from sklearn.cross_validation import train_test_split
import numpy as np
import scipy as sc
from scipy.io.wavfile import read
import scipy.signal

load_path='dataset/train/angry/' 
load_path2='./dataset/train/angry' 
save_path='./trainAngry.pkl' 
class_list1=[1]*15
class_list2=[0]*105

data = []
filenames = []
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
       print(item)
       nparr=sc.signal.resample(arr,1024).reshape(1,32,32)
       print(np.size(nparr))
       #arrp=sc.signal.resample(nparr,32)
       dat=nparr.astype(int)
       np.set_printoptions(threshold=np.nan)
       #print(dat)
       #print dat.shape
       data= np.array(dat)
       print(np.shape(data))
       print
       dodo12.append(data)

out_file = open(save_path, 'w+')
dic = {'data':dodo12}
cPickle.dump(dic, out_file)
out_file.close()

       
'''     
data2 = []
filenames2 = []
file_list2 = os.listdir(load_path2)
#print file_list2
d="dataset/train/notAngry/"
for item in  file_list2:
   if item.endswith(".wav"):
      n = os.path.join(load_path2, item)
      input = read(n)
      arr = np.array(input[1], dtype=float)
      nparr=sc.signal.resample(arr,1024) 
      dat=nparr.astype(int)
      #print dat
      filenames.append(dat)
      data2 = np.array(dat)

rotDat=np.concatenate((data,data2),axis=0)  
#print rotDat
#class_list=np.concatenate((class_list1, class_list2), axis=0)
out_file = open(save_path, 'w+')
dic = {'batch_label':'batch 1 of 1', 'data':rotDat, 'labels':class_list1+class_list2, 'filenames':filenames}
cPickle.dump(dic, out_file)
out_file.close()
'''

