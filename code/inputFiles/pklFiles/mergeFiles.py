import numpy as np
import cPickle


dat=np.load('trainAngryData.pkl')
print(np.shape(dat['data']))
'''
dat2=np.load('trainNotAngryData.pkl')
print(np.shape(dat2['data']))

dat3=np.concatenate((dat['data'],dat2['data']))
save_path='./trainData.pkl'
out_file = open(save_path, 'w+')
dic = {'data':dat3}
cPickle.dump(dic, out_file)
out_file.close()


dat23=np.load('./trainData.pkl')
print(np.shape(dat23['data']))
'''
#X_train=dat['data']
#y_train=dat['labels']
#dat2=np.load('test.pkl');
#X_test=dat2['data']
#y_test=dat2['labels']
#np.set_printoptions(threshold='nan')

