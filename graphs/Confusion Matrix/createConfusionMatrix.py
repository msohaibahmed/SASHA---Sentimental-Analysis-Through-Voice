print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
import cPickle
from sklearn import svm, datasets
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix

# import some data to play with
train_file_data = open('./graphs/trainData.pkl', 'r')
dat=cPickle.load(train_file_data)
X22=np.asarray(dat['data'])
train_file_data.close()
X_train=X22#np.squeeze(X22, axis=(2,)).shape
train_file_labels = open('./graphs/trainLabels.pkl', 'r')
dat2=cPickle.load(train_file_labels)
Y_train=dat2['labels']
train_file_labels.close()
y_train=np.asarray(Y_train)
test_file_data = open('./graphs/testData.pkl', 'r')
dat22=cPickle.load(test_file_data)
X_test=np.asarray(dat22['data'])
test_file_data.close()
test_file_label = open('./graphs/testLabels.pkl', 'r')
dat223=cPickle.load(test_file_label)
y_test=np.asarray(dat223['labels'])
test_file_label.close()
# Split the data into a training set and a test set
target_names=['angry','not angry']

# Run classifier, using a model that is too regularized (C too low) to see
# the impact on the results
classifier = svm.SVC(kernel='linear', C=0.01)

print(X_train.shape)
y_pred = classifier.fit(X_train, y_train).predict(X_test)


def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)
print('Confusion matrix, without normalization')
print(cm)
plt.figure()
plot_confusion_matrix(cm)
plt.savefig('confMatrix1.png')

# Normalize the confusion matrix by row (i.e by the number of samples
# in each class)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print('Normalized confusion matrix')
print(cm_normalized)
plt.figure()
plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')
plt.savefig('confMatrix.png')
plt.show()
