"""
finetuning Alexnet helper class
Must have a directory inside working directory titled alexnet_tune
with subfolders titled, filewriter, checkpoint
"""
import os
import numpy as np
import tensorflow as tf
from datetime import datetime
from alexnet import AlexNet
from datagenerator import ImageDataGenerator
from freqweight_all import calculateWeights

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allocator_type = 'BFC'

frequencyWeights = np.asarray(calculateWeights())

sparsityWeight = 600.

#training and testing paths
train_file = 'train_all.txt'
val_file = 'test_all.txt'

# learning params
learning_rate = 0.002
num_epochs = 50
batch_size = 200

# network params
dropout_rate = 0.5
num_classes = 2506
train_layers = ['fc8_new', 'fc7_1', 'fc7_2']
#train_layers = ['fc8', 'fc7', 'fc6']

display_step = 1

# path for the tf.summary.FileWriter and storing model checkpoints
cwd = os.getcwd()
filewriter_path = os.path.join(cwd, "alexnet_tune/filewriter")
checkpoint_path = os.path.join(cwd, "alexnet_tune/checkpoint")

# Create parent path if it doesn't exist
if not os.path.isdir(checkpoint_path): os.mkdir(checkpoint_path)

# TF placeholder for graph input and output
x = tf.placeholder(tf.float32, [batch_size, 227, 227, 3])
y = tf.placeholder(tf.float32, [None, num_classes])
keep_prob = tf.placeholder(tf.float32)

# Initialize model
model = AlexNet(x, keep_prob, num_classes, train_layers)

# Link variable to model output
# rounding only when over 0.6
score = model.fc8

# change the threshold in order to graph functions
pred85 = tf.round(tf.sigmoid(score) - 0.35)

# list of trainable variables of the layers we want to train
var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]

# GPU functions
with tf.device("/gpu:0"):
	# OP for calculating the loss
	with tf.name_scope("cross_ent"):

		crossEnt = tf.nn.weighted_cross_entropy_with_logits(targets = y, logits = score, pos_weight = sparsityWeight)
		loss = tf.reduce_mean(tf.multiply(crossEnt, frequencyWeights))
	
	with tf.name_scope("train"):
		# gradients of all trainable variables
		gradients = tf.gradients(loss, var_list)
		gradients = list(zip(gradients, var_list))
		
		# creating optimizer and using gradient descent
		optimizer = tf.train.GradientDescentOptimizer(learning_rate)
		train_op = optimizer.apply_gradients(grads_and_vars = gradients)
		
	with tf.name_scope("accuracy85"):
		tp85 = tf.count_nonzero(pred85 * y)
		tn85 = tf.count_nonzero((pred85 - 1) * (y - 1))
		fn85 = tf.count_nonzero((pred85 - 1) * y)
		fp85 = tf.count_nonzero(pred85 * (y - 1))
		accuracy85 = tf.divide(tf.add(tp85, tn85), tf.add(tf.add(tp85, fn85), tf.add(fp85, tn85)))
		precision85 = tf.divide(tp85, tf.add(tp85, fp85))
		recall85 = tf.divide(tp85, tf.add(tp85, fn85))
		f85 = 2 * tf.divide(tf.multiply(precision85, recall85), tf.add(precision85, recall85))
	init = tf.global_variables_initializer()
		
# inputting values into summary
for gradient, var in gradients:
	tf.summary.histogram(var.name + '/gradient', gradient)
for var in var_list:
	tf.summary.histogram(var.name, var)
tf.summary.scalar('cross_entropy', loss)
tf.summary.scalar('precision85', precision85)
tf.summary.scalar('recall85', recall85)
tf.summary.scalar('f85', f85)

# merging all summaries together
merged_summary = tf.summary.merge_all()

# initializing filewriter and saver for model ckpts
writer = tf.summary.FileWriter(filewriter_path)
saver = tf.train.Saver()

# Initalize the data generator seperately for the training and validation set
train_generator = ImageDataGenerator(train_file)
val_generator = ImageDataGenerator(val_file)

# Get the number of training/validation steps per epoch
train_batches_per_epoch = np.floor(train_generator.data_size / batch_size).astype(np.int16)
val_batches_per_epoch = np.floor(val_generator.data_size / batch_size).astype(np.int16)


with tf.Session(config = config) as sess:
	# initializing all variables
	sess.run(init)

	# add model graph to TensorBoard
	writer.add_graph(sess.graph)
	
	# Loading in the pretrained weights
	model.load_initial_weights(sess)
	
	print("{} Start training...".format(datetime.now()))
	print("{} Open Tensorboard at --logdir {}".format(datetime.now(), 
                                                    filewriter_path))

	# looping over epochs
	for epoch in range(num_epochs):
	
		print ("{} Epoch number: {}".format(datetime.now(), epoch+1))
		step = 1

		train_acc = 0.
		train_rec = 0.
		train_pre = 0.
		train_cost = 0.
		train_count = 0

		while step < train_batches_per_epoch:

			batch_xs, batch_ys = train_generator.next_batch(batch_size)

			#running the train op
			sess.run(train_op, feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout_rate})
			accTr = sess.run(accuracy85, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
			recTr = sess.run(recall85, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
			preTr = sess.run(precision85, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
			costTr = sess.run(loss, feed_dict = {x: batch_xs, y: batch_ys, keep_prob: 1.})

			train_cost += costTr
			train_acc += accTr
			train_rec += recTr
			train_pre += preTr
			train_count += 1

			#Generate summary and writing
			if step % display_step == 0:
				s = sess.run(merged_summary, feed_dict = {x: batch_xs, 
					y: batch_ys, keep_prob: 1.})
				writer.add_summary(s, epoch*train_batches_per_epoch + step)

			step += 1
		train_acc /= train_count
		train_rec /= train_count
		train_pre /= train_count
		train_cost /= train_count
		print("{} Training Accuracy = {:.4f}".format(datetime.now(), train_acc))
		print("{} Training Recall = {:.4f}".format(datetime.now(), train_rec))
		print("{} Training Precision = {:.4f}".format(datetime.now(), train_pre))
		print("{} Training Cost = {:.4f}".format(datetime.now(), train_cost))
		print("{} Start validation".format(datetime.now()))

		test_acc = 0.
		test_rec = 0.
		test_pre = 0.
		test_cost = 0.
		test_count = 0

		for _ in range(val_batches_per_epoch):
			batch_tx, batch_ty = val_generator.next_batch(batch_size)
			accTe = sess.run(accuracy85, feed_dict={x: batch_tx, y: batch_ty, keep_prob: 1.})
			recTe = sess.run(recall85, feed_dict={x: batch_tx, y: batch_ty, keep_prob: 1.})
			preTe = sess.run(precision85, feed_dict={x: batch_tx, y: batch_ty, keep_prob: 1.})
			costTe = sess.run(loss, feed_dict = {x: batch_tx, y: batch_ty, keep_prob: 1.})

			test_cost += costTe
			test_acc += accTe
			test_rec += recTe
			test_pre += preTe
			test_count += 1
		test_acc /= test_count
		test_rec /= test_count
		test_pre /= test_count
		test_cost /= test_count
		print("{} Test Accuracy = {:.4f}".format(datetime.now(), test_acc))
		print("{} Test Recall = {:.4f}".format(datetime.now(), test_rec))
		print("{} Test Precision = {:.4f}".format(datetime.now(), test_pre))
		print("{} Test Cost = {:.4f}".format(datetime.now(), test_cost))

		val_generator.reset_pointer()
		train_generator.reset_pointer()


		if epoch == num_epochs - 1:
			print("{} Saving checkpoint of model...".format(datetime.now()))

			#save checkpoint of the model
			checkpoint_name = os.path.join(checkpoint_path, 'my_model')
			save_path = saver.save(sess, checkpoint_name)

			print("{} Model checkpoint saved at {}".format(datetime.now(), checkpoint_name))
