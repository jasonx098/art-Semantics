import numpy as np
from scipy.misc import imread, imresize

"""
forms a data generator that can produce the next batch of
testing or training images along with their labels
"""

class ImageDataGenerator:
	def __init__(self, filepath, outputLen = 2506, imgsize = (227, 227, 3),
	mean = np.array([133., 120., 103.])):
		self.SIZE = imgsize
		self.OUTPUT = outputLen
		# points to where in each list
		self.POINTER = 0
		self.MEAN = mean
		self.readfilepath(filepath)
	
	def readfilepath(self, filepath):
		# reads in the file path
		with open(filepath) as f:
			self.images = []
			self.labels = []
			# creates list of image paths
			# creates list of labels to corresponding image
			for line in f:
				items = line.split(',')
				self.images.append(items[0])
				
				indices = []
				for index in items[1].strip().split():
					indices.append(int(index))
				self.labels.append(indices)
			self.data_size = len(self.labels)
				
	def reset_pointer(self):
		# sets pointer to front of list
		self.POINTER = 0
		
	def next_batch(self, batch_size):
		# function returns the next batch of images as a np array of 4d size
		# returns the labels in a numpy array as well
		
		impaths = self.images[self.POINTER:self.POINTER + batch_size]
		imlabs = self.labels[self.POINTER:self.POINTER + batch_size]
		
		self.POINTER += batch_size
		
		# creating an N-D array first and then filling in values
		imgs = np.ndarray([batch_size, self.SIZE[0], self.SIZE[1], self.SIZE[2]])
		for i in range(len(impaths)):
			currIm = imread(impaths[i])

			currIm = imresize(currIm, self.SIZE)
			currIm = currIm.astype(np.float32)
			# mean subtraction for each channel
			imgs[i] = currIm - self.MEAN
		
		labels = np.zeros((batch_size, self.OUTPUT))
		for i in range(len(imlabs)):
			labels[i][imlabs[i]] = 1
		
		# return final outputs
		return imgs, labels
				
"""
# testing
gen = ImageDataGenerator('training.txt', 2506)
a = gen.next_batch(50)
print "img shape is %s, label shape is %s" % (a[0].shape, a[1].shape)

img shape is (50, 227, 227, 3), label shape is (50, 2506)
"""

