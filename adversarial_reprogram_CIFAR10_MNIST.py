import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np

import tensorflow as tf
from PIL import Image
import tensorflow.contrib.slim as slim
import time
from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image


size = 299
batch_size = 50
starter_learning_rate = 0.01
decay_rate = 0.96
num_batches = int(55000/ batch_size)
decay_steps = 2* num_batches
reg_lambda = 0.005
display_step = 100
total_steps = 6

output_mapping = np.zeros([10, 10])
output_mapping[0:10, 0:10] = np.eye(10)

input_mask = np.pad(np.zeros([1, 28, 28, 3]),[[0,0], [136, 135], [136, 135], [0,0]],'constant', constant_values = 1)

def preprocess(img) :
	return (img - 0.5)*2.0

def deprocess(img) :
	return (img +1)/2.0	

with tf.Graph().as_default():
	global_step = tf.Variable(0, trainable=False)
	# mask = tf.pad(tf.constant(np.zeros([1, 28, 28, 3]), dtype = tf.float32), 
	# 		paddings = tf.constant([[0,0], [136, 135], [136, 135], [0,0]]), constant_values=1)
	mask = tf.constant(input_mask, dtype = tf.float32)
	
	weights = tf.get_variable('adv_weight', shape = [1, size, size, 3], dtype = tf.float32)
	input_image = tf.placeholder(shape = [None, 28,28,1], dtype = tf.float32)
	channel_image = tf.concat([input_image, input_image, input_image], axis = -1)
	rgb_image = tf.pad(tf.concat([input_image, input_image, input_image], axis = -1), 
				paddings = tf.constant([[0,0], [136, 135], [136, 135], [0,0]]))

	adv_image = tf.nn.tanh(tf.multiply(weights, mask)) + rgb_image
	print("\n shape\n")
	print(adv_image.get_shape())

	labels = tf.placeholder(tf.float32, shape=[None, 10])
	
	from cleverhans.model import Model
	#from cleverhans.model_zoo.basic_cnn import ModelBasicCNN
	from substitute.substitute_model_sceleton import ModelBasicCNN

	# Define TF model graph (for the black-box model)
	nb_filters = 64
	nb_classes = 10
	model = ModelBasicCNN('model1', nb_classes, nb_filters)
	logits = model.get_logits(adv_image)

	# debug
	print("\noriginal logits")
	print(logits)
	total_logits = tf.reduce_sum(logits)
	total_logits = tf.Print(total_logits, [total_logits], 'total_logits')
	logits = tf.Print(logits, [logits], 'logits')

	print("Defined TensorFlow model graph.")

	#logits,_ = tf.contrib.slim.nets.inception.inception_v3(adv_image,num_classes = 1001,is_training=False)
	
	output_mapping_tensor = tf.constant(output_mapping, dtype = tf.float32)

	# debug
	print("\noutput_mapping_tensor", output_mapping_tensor)
	with tf.Session() as sess:
		result = sess.run(output_mapping_tensor)
		print(result)


	new_logits = tf.matmul(logits, output_mapping_tensor)

	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = new_logits, labels = labels)) + reg_lambda * tf.nn.l2_loss(weights)

	learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
										   decay_steps, decay_rate, staircase=True)

	correct_prediction = tf.equal(tf.argmax(new_logits,1), tf.argmax(labels,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step = global_step, var_list = [weights])
	
	'''
	variables_to_restore = slim.get_model_variables('InceptionV3')
	sess = tf.InteractiveSession()
	sess.run(tf.global_variables_initializer())
	saver = tf.train.Saver(variables_to_restore)
	saver1 = tf.train.Saver([weights])

	saver.restore(sess,'./inception_v3.ckpt')
	'''

	# restore if checkpoint flies exist
	#tf.reset_default_graph()
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())

	imported_meta = tf.train.import_meta_graph('./substitute/saved_model.ckpt.meta')
	saver1 = tf.train.Saver([weights])
	imported_meta.restore(sess, tf.train.latest_checkpoint('./substitute'))
	

	#ckpt = tf.train.get_checkpoint_state("./saved_model.ckpt")
	#saver.restore(sess,ckpt.model_checkpoint_path)
	print("Restored model")
	
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)  
for i in range(total_steps) :
	batch = mnist.train.next_batch(batch_size)
	_, l, acc = sess.run([train_step, loss, accuracy], feed_dict = {input_image : preprocess(np.reshape(batch[0], [-1, 28, 28, 1])),
 									labels : batch[1]})
	if i%display_step == 0 :
		print('after %d steps the loss is %g and train_acc is %g'%(i, l, acc))
	# debug
	print("\ntotal logits\n")
	print(total_logits)
	total_logits = tf.Print(total_logits, [total_logits], 'total_logits')


saver1.save(sess, './adversarial_model/adversarial_reprogramming_model.ckpt')

test_iterations = int(len(mnist.test.images)/ batch_size)
total_correct = 0
for i in range(test_iterations) :
 	test_imgs = preprocess(np.reshape(mnist.test.images[i*batch_size:(i+1)*batch_size], [-1,28,28,1]))
 	test_labels = mnist.test.labels[i*batch_size:(i+1)*batch_size]
 	corrects = sess.run(accuracy, feed_dict = {input_image : test_imgs, labels : test_labels})
 	total_correct += corrects * batch_size

print('test acc is ', float(total_correct)/len(mnist.test.images))



		
				
				        
	   

