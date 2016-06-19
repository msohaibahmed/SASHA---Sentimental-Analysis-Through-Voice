import numpy as np
import cPickle


dat=np.load('./pkl/dataTrainAngry.pkl')
print(np.shape(dat['data']))

dat2=np.load('./pkl/dataTrainNotAngry.pkl')
print(np.shape(dat2['data']))
'''
dat23=np.load('./testData.pkl')
print(np.shape(dat23['data']))
'''
dat3=np.concatenate((dat['data'],dat2['data']))
save_path='./trainData.pkl'
out_file = open(save_path, 'w+')
dic = {'data':dat3}
cPickle.dump(dic, out_file)
out_file.close()

