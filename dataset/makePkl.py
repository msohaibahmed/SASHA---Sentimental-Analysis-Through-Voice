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

load_path='./savee/train/notAngry/' 
save_path='./test_label.pkl' 
class_label1=[1]*12
class_label2=[0]*105
'''
data = []
filenames = []
file_list = os.listdir(load_path)
d="dataset/train/angry/"
i=1
total_item=1
dodo12=[]	
print("Savee Not Angry")
for item in  file_list:
    if item.endswith(".wav"):
       print item,i
       i=i+1
       total_item+=1
       n = os.path.join(load_path, item)
       input = read(n)
       arr = np.array(input[1], dtype=float)
       nparr=sc.signal.resample(arr,1024).reshape(1,32,32)
       #arrp=sc.signal.resample(nparr,32)
       dat=nparr.astype(int)
       np.set_printoptions(threshold=np.nan)
       #print(dat)
       #print dat.shape
       filenames.append(n)
       data = np.array(dat)
       dodo12.append(data)

out_file = open(save_path, 'w+')
dic = {'data':dodo12}
cPickle.dump(dic, out_file)
out_file.close()


load_path='./enterface/angry/' 
save_path='./enterface_train_angry_data.pkl' 
class_list1=[1]*45
class_list2=[0]*285

data = []
filenames = []
file_list = os.listdir(load_path)
d="dataset/train/angry/"
i=1
dodo12=[]	
print("Enterface Angry")
for item in  file_list:
    if item.endswith(".wav"):
       print item,i
       i=i+1
       total_item+=1
       n = os.path.join(load_path, item)
       input = read(n)
       arr = np.array(input[1], dtype=float)
       nparr=sc.signal.resample(arr,512).reshape(1,32,32)
       #arrp=sc.signal.resample(nparr,32)
       dat=nparr.astype(int)
       np.set_printoptions(threshold=np.nan)
       #print(dat)
       #print dat.shape
       filenames.append(n)
       data = np.array(dat)
       dodo12.append(data)

out_file = open(save_path, 'w+')
dic = {'data':dodo12}
cPickle.dump(dic, out_file)
out_file.close()

print("Enterface Angry Done")


savee_angry=np.load('./train_angry_data.pkl')
print(np.shape(savee_angry['data']))

enterface_angry=np.load('./savee_train_not_angry_data.pkl')
print(np.shape(enterface_angry['data']))

conc_data=np.concatenate((savee_angry['data'],enterface_angry['data']))
save_path='./train_data.pkl'
out_file = open(save_path, 'w+')
dic = {'data':conc_data}
cPickle.dump(dic, out_file)
out_file.close()

enterface_angry=np.load('./train_data.pkl')
print(np.shape(enterface_angry['data']))
'''

out_file = open(save_path, 'w+')
dic = {'labels':class_label1+class_label2}
cPickle.dump(dic, out_file)
out_file.close()



