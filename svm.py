
import numpy
def convertManyToOne(Y):
	newY = numpy.empty((0, 1))
	for i in xrange(len(Y)):
		for j in xrange(len(Y[i])):
			if Y[i][j] == 1:
				newY = numpy.vstack([newY, j])
				break
	return newY

import scipy.io as sio

X = sio.loadmat('X_scaled.mat')['X_scaled']
Y = sio.loadmat('Y.mat')['Y']
Y = convertManyToOne(Y)
Y = numpy.htsack(Y)
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
from sklearn.svm import SVC
clf = SVC(verbose=True, max_iter=10, cache_size = 500)
clf.fit(X_train, y_train)
