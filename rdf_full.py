import numpy
def convertManyToOne(Y):
	newY = numpy.empty((0, 1))
	for i in xrange(len(Y)):
		for j in xrange(len(Y[i])):
			if Y[i][j] == 1:
				newY = numpy.vstack([newY, j])
				break
	return newY

num_emotions = 7
featuresPerOutputClass = 4#min, max, avg
featuresPerFile = num_emotions*featuresPerOutputClass + 1 #+1 for audio id
probailityThreshold = 0.2
import scipy.io as sio
import pickle
X = sio.loadmat('X_scaled.mat')['X_scaled']
Y = sio.loadmat('Y.mat')['Y']
Y = convertManyToOne(Y)
Y = numpy.hstack(Y)

def getYValues(X, map):
	audio_ids = X[:, -1]
	Y = numpy.empty((0, 1))
	for cur_id in audio_ids:
		emotion = map[cur_id]
		Y = numpy.vstack([Y, emotion])
	return Y


#returns utterance level features, last column is audio id
def getUtteranceLevelFeatures(clf, X, ids):
	probabilities = clf.predict_proba(X)
	utterances = {} #each will map from id to 2d numpy array of probabilities
	#each row is probability values for a single segment 
	for i in xrange(len(X)):
		cur_id = ids[i]
		prob = probabilities[i]
		if not utterances.has_key(cur_id):
			utterances[cur_id] = numpy.empty((0, len(prob)))
		utterances[cur_id] = numpy.vstack([utterances[cur_id], prob])

	#for each output class, we have these features - avg of probabilities, min, max
	audio_features = numpy.empty((0, featuresPerFile))

	for audio_id in utterances:
		prob_2d = utterances[audio_id]
		cur_features = numpy.array([])
		for emo_id in xrange(num_emotions):
			prob_emo = prob_2d[:, emo_id]
			cur_features = numpy.append(cur_features, numpy.max(prob_emo))
			cur_features = numpy.append(cur_features, numpy.min(prob_emo))
			cur_features = numpy.append(cur_features, numpy.mean(prob_emo))
			cur_features = numpy.append(cur_features, numpy.mean(prob_emo>=probailityThreshold))
		cur_features = numpy.append(cur_features, audio_id)
		audio_features = numpy.vstack([audio_features, cur_features])
	return audio_features


from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

X_train_ids = X_train[:, -1]
X_test_ids = X_test[:, -1]
X_train = X_train[:, 0:-1]
X_test = X_test[:, 0:-1]

from sklearn.ensemble import RandomForestClassifier
clf =  RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train) #exlude last column it has column ids

print "Training Accuracy %.2f\n" % (clf.score(X_train, y_train))
print "Testing Accuracy %.2f\n" % (clf.score(X_test, y_test))


#start building utterance level classifier
newTrainX = getUtteranceLevelFeatures(clf, X_train, X_train_ids)
newTestX = getUtteranceLevelFeatures(clf, X_test, X_test_ids)
id_to_emotion_map = pickle.load( open( "id_to_emotion.p", "rb" ) )
newTrainY = getYValues(newTrainX, id_to_emotion_map)
newTestY = getYValues(newTestX, id_to_emotion_map)
sio.savemat('newTrainX.mat', {'newTrainX' : newTrainX})
sio.savemat('newTrainY.mat', {'newTrainY' : newTrainY})
sio.savemat('newTestX.mat', {'newTestX' : newTestX})
sio.savemat('newTestY.mat', {'newTestY' : newTestY})
utteranceClf =  RandomForestClassifier(n_estimators=100)
utteranceClf.fit(newTrainX, numpy.hstack(newTrainY))
print "Training Done\n"
print "Training Accuracy %.2f\n" % (utteranceClf.score(newTrainX, newTrainY))
print "Testing Accuracy %.2f\n" % (utteranceClf.score(newTestX, newTestY))
