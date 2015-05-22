import essentia
import essentia.standard
import os
import scipy.io.wavfile
import scipy.io
from essentia.standard import *
import numpy
from sklearn import preprocessing

numpy.seterr(all='warn')


class Segment:
	def __init__(self, segment_features, segment_energy, output):
		self.segment_features = segment_features
		self.segment_energy = segment_energy
		self.output = output

def getSegment(frames, start, end):
	rows = frames[start : end+1, ]
	return numpy.hstack(rows)

def convertManyToOne(Y):
	newY = numpy.empty((0, 1))
	for i in xrange(len(Y)):
		for j in xrange(len(Y[i])):
			if Y[i][j] == 1:
				newY = numpy.vstack([newY, j])
				break
	return newY

def filterSegments(segments, threshold):
	#sort segments by energy
	segments.sort(key = lambda segment : segment.segment_energy)
	num_segments = len(segments)
	threshold_segment_idx = int(num_segments*threshold)
	threshold_energy = segments[threshold_segment_idx].segment_energy
	#filter list
	return [segment for segment in segments if segment.segment_energy >= threshold_energy]

folder_path = '/home/karan173/Desktop/btp/berlin_db/wav/'
audios = []
emotions = []
emotion_class_map = {'W' : 0, 'L' : 1, 'E' : 2, 'A' : 3, 'F' : 4, 'T' : 5, 'N' : 6}
num_emotions = len(emotion_class_map)

for filename in os.listdir(folder_path):
	filepath = folder_path + filename
	# print filepath
	#rate, data = scipy.io.wavfile.read(filename = filepath)
	loader = essentia.standard.MonoLoader(filename = filepath)
	data = loader()
	emotion = filepath[-6] #2nd last character of file exluding extension name wav
	emotion_class = emotion_class_map[emotion]
	audios.append(data)
	emotions.append(emotion_class)

sample_rate = 16000 # in hertz
frameDuration = 0.025 #duration in seconds
hopDuration = 0.010
frameSize = int(sample_rate*frameDuration)
hopSize = int(sample_rate*hopDuration)
featuresPerFrame = 13 + 1
framesPerSegment = 25
featuresPerSegment = featuresPerFrame * framesPerSegment
segmentHop = 13
energyPercentThreshold = 0.20 #fractional
w = Windowing(type = 'hann')
spectrum = Spectrum()
mfcc = MFCC()


#will map from audio number to list of Segments
utterances = {}

		
X = numpy.empty((0, featuresPerSegment))
Y = numpy.empty((0, num_emotions))
energy_func = essentia.standard.Energy()
audio_ids = []

for i in range(len(audios)):
	print i
	utterances[i] = []
	audio = audios[i]
	output = emotions[i]
	output_vec = numpy.zeros((1, num_emotions))
	output_vec[0][output] = 1
	frames = numpy.empty((featuresPerFrame, )) #each index stores a list of 53 features for that frame
	# frames = numpy.matrix(frames)
	#for frame in FrameGenerator(essentia.array(audio), frameSize = frameSize, hopSize = hopSize):
	for frame in FrameGenerator(audio, frameSize = frameSize, hopSize = hopSize):
		mfcc_bands, mfcc_coeffs = mfcc(spectrum(w(frame))) #40 mfcc bands and 13 mfcc_coeffs
		# frame_features = numpy.concatenate((mfcc_bands,mfcc_coeffs))
		frame_energy = energy_func(audio)
		frame_features = numpy.append(mfcc_coeffs, frame_energy)
		if numpy.isnan(frame_features).any() :
			print "nan\nnan\n"
			exit()
		frames = numpy.vstack([frames,frame_features])

	start_segment = 0
	while True:
		end_segment = start_segment + framesPerSegment - 1 #center = 13, left=1-12, right=14-25
		if end_segment >= len(frames) :
			break
		segment = getSegment(frames, start_segment, end_segment)
		segment_energy = energy_func(essentia.array(segment))
		start_segment = start_segment + segmentHop -1 #segmentSize = 13
		segment_obj = Segment(segment, segment_energy, output_vec) 
		utterances[i].append(segment_obj)
	utterances[i] = filterSegments(utterances[i], energyPercentThreshold)

	for segment in utterances[i]:
		segment_features = numpy.append(segment.segment_features, segment.segment_energy)
		X = numpy.vstack([X, segment.segment_features])
		Y = numpy.vstack([Y, segment.output])
		audio_ids.append(i)

X_scaled = preprocessing.scale(X)
scipy.io.savemat('Y.mat', {'Y' : Y})
scipy.io.savemat('X_scaled.mat', {'X_scaled' : X_scaled})
scipy.io.savemat('audio_ids.mat', {'audio_ids' : audio_ids})
