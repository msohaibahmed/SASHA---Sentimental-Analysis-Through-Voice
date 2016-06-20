import numpy as np
import cPickle


dat=np.load('./savee_test_angry_data.pkl')
print(np.shape(dat['data']))

dat2=np.load('./savee_test_not_angry_data.pkl')
print(np.shape(dat2['data']))

dat3=np.concatenate((dat['data'],dat2['data']))
save_path='./test_data.pkl'
out_file = open(save_path, 'w+')
dic = {'data':dat3}
cPickle.dump(dic, out_file)
out_file.close()



dat23=np.load('./test_data.pkl')
print(np.shape(dat23['data']))
'''
#X_train=dat['data']
#y_train=dat['labels']
#dat2=np.load('test.pkl');
#X_test=dat2['data']
#y_test=dat2['labels']
#np.set_printoptions(threshold='nan')
'''

