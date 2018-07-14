import tensorflow as tf
import numpy as np
"""
Alexnet Architecture 
"""
class AlexNet(object):
	
	def __init__(self, x, keep_prob, num_classes, skip_layer,
			 weights_path = 'DEFAULT'):
		# instance variables
		self.X = x
		self.NUM_CLASSES = num_classes
		self.KEEP_PROB = keep_prob
		self.SKIP_LAYER = skip_layer
		
		if weights_path == 'DEFAULT':
			self.WEIGHTS_PATH = 'my2fcmodel.npy'
			#self.WEIGHTS_PATH = 'mynet.npy'
			#self.WEIGHTS_PATH = 'bvlc_alexnet.npy'
		else:
			self.WEIGHTS_PATH = weights_path
			
		self.create()
	
	def create(self):
		with tf.device("/gpu:0"):
			
			#1st layer: CONV, RELU, POOL, LRN
			conv1 = conv(self.X, 11, 11, 96, 4, 4, padding = 'VALID', name = 'conv1')
			pool1 = max_pool(conv1, 3, 3, 2, 2, padding = 'VALID', name = 'pool1')
			norm1 = lrn(pool1, 2, 2e-05, 0.75, name = 'norm1')

			#2nd layer: CONV, RELU, POOL, LRN 2 groups
			conv2 = conv(norm1, 5, 5, 256, 1, 1, groups = 2, name = 'conv2')
			pool2 = max_pool(conv2, 3, 3, 2, 2, padding = 'VALID', name ='pool2')
			norm2 = lrn(pool2, 2, 2e-05, 0.75, name = 'norm2')
			
			#3rd Layer: CONV RELU
			conv3 = conv(norm2, 3, 3, 384, 1, 1, name = 'conv3')
			
			#4th LAYER: CONV RELU 2 groups
			conv4 = conv(conv3, 3, 3, 384, 1, 1, groups = 2, name = 'conv4')
			
			#5th Layer: CONV RELU POOL 2 groups
			conv5 = conv(conv4, 3, 3, 256, 1, 1, groups = 2, name = 'conv5')
			pool5 = max_pool(conv5, 3, 3, 2, 2, padding = 'VALID', name = 'pool5')

			#6th layer: Flatten, FC RELU, Dropout
			flattened = tf.reshape(pool5, [-1, 6*6*256])
			fc6 = fc(flattened, 6*6*256, 4096, name='fc6')
			dropout6 = dropout(fc6, self.KEEP_PROB)

			"""
			DEPENDING ON USING WHICH MODEL
			#7th Layer: FC RELU, DROPOUT
			fc7 = fc(dropout6, 4096, 4096, name = 'fc7')
			dropout7 = dropout(fc7, self.KEEP_PROB)
			
			#8th Layer: FC, leave sigmoid for finetuning
			self.fc8 = fc(dropout7, 4096, self.NUM_CLASSES, relu = False, name='fc8')
			"""

			#7th Layer: FC RELU, DROPOUT
			fc7 = fc(dropout6, 4096, 4096, name = 'fc7')
			dropout7 = dropout(fc7, self.KEEP_PROB)

			fc7_1 = fc(dropout7, 4096, 1024, name = 'fc7_1')
			dropout7_1 = dropout(fc7_1, self.KEEP_PROB)

			fc7_2 = fc(dropout7_1, 1024, 512, name = 'fc7_2')
			dropout7_2 = dropout(fc7_2, self.KEEP_PROB)
			#8th Layer: FC, leave sigmoid for finetuning
			self.fc8 = fc(dropout7_2, 512, self.NUM_CLASSES, relu = False, name='fc8_new')
			
	def load_initial_weights(self, session):
		
		weights_dict = np.load(self.WEIGHTS_PATH, encoding = 'bytes').item()
		
		# loopingover all layer names
		for op_name in weights_dict:
			
			# check if layer is not in skip layer
			if op_name not in self.SKIP_LAYER:
			
				with tf.variable_scope(op_name, reuse = True):
					
					for data in weights_dict[op_name].values():
					
						# for biases
						if len(data.shape) == 1:
							var = tf.get_variable('biases', trainable = False)
							session.run(var.assign(data))
						# for weights
						else:
							var = tf.get_variable('weights', trainable = False)
							session.run(var.assign(data))
	
"""
predefining all layers for Alexnet
"""
	
def conv(x, filter_height, filter_width, num_filters, stride_y, stride_x, name, 
	padding = 'SAME', groups = 1):
	
	# input channels
	input_channels = int(x.get_shape()[-1])
	
	# create lambda function for convolution
	convolve = lambda i, k: tf.nn.conv2d(i, k, strides = [1, stride_y, stride_x, 1],
			padding = padding)
			
	with tf.variable_scope(name) as scope:
		# creates tf variables for the weights and biases of the conv layer
		weights = tf.get_variable('weights', shape = [filter_height, filter_width, 
			input_channels/groups, num_filters])
		biases = tf.get_variable('biases', shape= [num_filters])
		
		if groups == 1:
			conv = convolve(x, weights)
		
		# multiple group cases
		else:
			# splitting input and weights separately
			input_groups = tf.split(axis = 3, num_or_size_splits=groups, value=x)
			weight_groups = tf.split(axis = 3, num_or_size_splits=groups, value=weights)
			output_groups = [convolve(i, k) for i,k in zip(input_groups, weight_groups)]
			
			conv = tf.concat(axis = 3, values = output_groups)
			
		# adding biases
		bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())
		
		# applying relu
		return tf.nn.relu(bias, name = scope.name)


def fc(x, num_in, num_out, name, relu = True):
	with tf.variable_scope(name) as scope:
	
		# creating tf variables for the weights and biases
		weights = tf.get_variable('weights', shape=[num_in, num_out], trainable = True)
		biases = tf.get_variable('biases', [num_out], trainable = True)
		
		# Matrix multiplication
		act = tf.nn.xw_plus_b(x, weights, biases, name = scope.name)
		
		if relu == True:
			# applying ReLu non linearity
			relu = tf.nn.relu(act)
			return relu
		else:
			return act

def max_pool(x, filter_height, filter_width, stride_y, stride_x, name, padding = 'SAME'):
	return tf.nn.max_pool(x, ksize= [1, filter_height, filter_width, 1], 
						strides = [1, stride_y, stride_x, 1], 
						padding = padding, name = name)

def lrn(x, radius, alpha, beta, name, bias = 1.0):
	return tf.nn.local_response_normalization(x, depth_radius = radius, alpha = alpha,
						beta = beta, bias = bias, name = name)

def dropout(x, keep_prob):
	return tf.nn.dropout(x, keep_prob)

