import os
import numpy as np
import tensorflow as tf
from datagenerator import ImageDataGenerator

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True;
top = 2506
classes = 2506
batch_size = 200
val_file = 'test_all.txt'

def conv(x, filter_height, filter_width, num_filters, stride_y, stride_x, w, b,
	padding = 'SAME', groups = 1):

	weights = w
	biases = b
	
	# input channels
	input_channels = int(x.get_shape()[-1])
	
	# create lambda function for convolution
	convolve = lambda i, k: tf.nn.conv2d(i, k, strides = [1, stride_y, stride_x, 1],
			padding = padding)

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
	return tf.nn.relu(bias)

def fc(x, num_in, num_out, w, b, relu = True):
	
	weights = w
	biases = b
	
	# Matrix multiplication
	act = tf.nn.xw_plus_b(x, weights, biases)
	
	if relu == True:
		# applying ReLu non linearity
		relu = tf.nn.relu(act)
		return relu
	else:
		return act

def max_pool(x, filter_height, filter_width, stride_y, stride_x, padding = 'SAME'):
	return tf.nn.max_pool(x, ksize= [1, filter_height, filter_width, 1], 
						strides = [1, stride_y, stride_x, 1], 
						padding = padding)

def lrn(x, radius, alpha, beta, bias = 1.0):
	return tf.nn.local_response_normalization(x, depth_radius = radius, alpha = alpha,
						beta = beta, bias = bias)

def dropout(x, keep_prob):
	return tf.nn.dropout(x, keep_prob)

def alex(X, KEEP_PROB, NUM_CLASSES):
		#1st layer: CONV, RELU, POOL, LRN
	conv1 = conv(X, 11, 11, 96, 4, 4, w = conv1w, b = conv1b, padding = 'VALID')
	pool1 = max_pool(conv1, 3, 3, 2, 2, padding = 'VALID')
	norm1 = lrn(pool1, 2, 2e-05, 0.75)
		
	#2nd layer: CONV, RELU, POOL, LRN 2 groups
	conv2 = conv(norm1, 5, 5, 256, 1, 1, w = conv2w, b = conv2b, groups = 2)		
	pool2 = max_pool(conv2, 3, 3, 2, 2, padding = 'VALID')
	norm2 = lrn(pool2, 2, 2e-05, 0.75)
			
	#3rd Layer: CONV RELU
	conv3 = conv(norm2, 3, 3, 384, 1, 1, w = conv3w, b = conv3b)
			
	#4th LAYER: CONV RELU 2 groups
	conv4 = conv(conv3, 3, 3, 384, 1, 1, w = conv4w, b = conv4b, groups = 2)
			
	#5th Layer: CONV RELU POOL 2 groups
	conv5 = conv(conv4, 3, 3, 256, 1, 1, w = conv5w, b = conv5b, groups = 2)
	pool5 = max_pool(conv5, 3, 3, 2, 2, padding = 'VALID')
			
	#6th layer: Flatten, FC RELU, Dropout
	flattened = tf.reshape(pool5, [-1, 6*6*256])
	fc6 = fc(flattened, 6*6*256, 4096, w = fc6w, b = fc6b)
	dropout6 = dropout(fc6, KEEP_PROB)
			
	#7th Layer: FC RELU, DROPOUT
	fc7 = fc(dropout6, 4096, 4096, w = fc7w, b = fc7b)
	dropout7 = dropout(fc7, KEEP_PROB)

	fc7_1 = fc(dropout7, 4096, 1024, w = fc7_1w, b = fc7_1b)
	dropout7_1 = dropout(fc7_1, KEEP_PROB)

	fc7_2 = fc(dropout7_1, 1024, 512, w = fc7_2w, b = fc7_2b)
	dropout7_2 = dropout(fc7_2, KEEP_PROB)
		
	#8th Layer: FC, leave sigmoid for finetuning
	fc8 = fc(dropout7_2, 512, NUM_CLASSES, w = fc8_neww, b = fc8_newb, relu = False)
	return fc8

val_generator = ImageDataGenerator(val_file)
val_batches_per_epoch = np.floor(val_generator.data_size / batch_size).astype(np.int16) + 1

with tf.device("/gpu:1"):
	with tf.Session(config = config) as sess:
		x = tf.placeholder(tf.float32, [batch_size, 227, 227, 3])
		keep_prob = tf.placeholder(tf.float32)

		saver = tf.train.import_meta_graph('alexnet_tune/checkpoint/my_model.meta')
		saver.restore(sess, tf.train.latest_checkpoint('alexnet_tune/checkpoint/'))

		graph = tf.get_default_graph()
		conv1w = graph.get_tensor_by_name("conv1/weights:0")
		conv1b = graph.get_tensor_by_name("conv1/biases:0")

		conv2w = graph.get_tensor_by_name("conv2/weights:0")
		conv2b = graph.get_tensor_by_name("conv2/biases:0")

		conv3w = graph.get_tensor_by_name("conv3/weights:0")
		conv3b = graph.get_tensor_by_name("conv3/biases:0")

		conv4w = graph.get_tensor_by_name("conv4/weights:0")
		conv4b = graph.get_tensor_by_name("conv4/biases:0")

		conv5w = graph.get_tensor_by_name("conv5/weights:0")
		conv5b = graph.get_tensor_by_name("conv5/biases:0")

		fc6w = graph.get_tensor_by_name("fc6/weights:0")
		fc6b = graph.get_tensor_by_name("fc6/biases:0")

		fc7w = graph.get_tensor_by_name("fc7/weights:0")
		fc7b = graph.get_tensor_by_name("fc7/biases:0")

		fc7_1w = graph.get_tensor_by_name("fc7_1/weights:0")
		fc7_1b = graph.get_tensor_by_name("fc7_1/biases:0")

		fc7_2w = graph.get_tensor_by_name("fc7_2/weights:0")
		fc7_2b = graph.get_tensor_by_name("fc7_2/biases:0")

		fc8_neww = graph.get_tensor_by_name("fc8_new/weights:0")
		fc8_newb = graph.get_tensor_by_name("fc8_new/biases:0")

		final = tf.sigmoid(alex(x, keep_prob, classes))
		val, ind = tf.nn.top_k(final, top)

		f = open('all_val.txt', 'w')
		g = open('all_ind.txt', 'w')
		for _ in range(val_batches_per_epoch):
			batch_tx, batch_ty = val_generator.next_batch(batch_size)

			values = sess.run(val, feed_dict={x:batch_tx, keep_prob:1.0})
			indices = sess.run(ind, feed_dict={x:batch_tx, keep_prob: 1.0})

			for i in range(values.shape[0]):
				for j in range(values.shape[1]):
					f.write(str(values[i, j]) + ' ')
					g.write(str(indices[i, j]) + ' ')
				f.write('\n')
				g.write('\n')
		f.close()
		g.close()




