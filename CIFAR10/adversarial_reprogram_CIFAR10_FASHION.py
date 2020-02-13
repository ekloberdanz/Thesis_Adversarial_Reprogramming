import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np

import tensorflow as tf
import tensorflow.contrib.slim as slim
import time
from tensorflow.examples.tutorials.mnist import input_data
import keras
#from PIL import Image



orig_image_size = 32
new_image_size = 28
batch_size = 50
starter_learning_rate = 0.01
decay_rate = 0.96
num_batches = int(55000/ batch_size)
decay_steps = 2* num_batches
reg_lambda = 0.005
display_step = 100
total_steps = 1000

output_mapping = np.zeros([10, 10])
output_mapping[0:10, 0:10] = np.eye(10)

input_mask = np.pad(np.zeros([1, new_image_size, new_image_size, 3]),\
	[[0,0], [int((np.ceil(orig_image_size/2.))-new_image_size/2.), int((np.floor(orig_image_size/2.))-new_image_size/2.)], \
	[int((np.ceil(orig_image_size/2.))-new_image_size/2.), int((np.floor(orig_image_size/2.))-new_image_size/2.)],\
	[0,0]],'constant', constant_values = 1)

def preprocess(img) :
	return (img - 0.5)*2.0

def deprocess(img) :
	return (img +1)/2.0	

with tf.Graph().as_default():
	# Set TF random seed to improve reproducibility
	tf.set_random_seed(34)

	global_step = tf.Variable(0, trainable=False)
	# mask = tf.pad(tf.constant(np.zeros([1, 28, 28, 3]), dtype = tf.float32), 
	# 		paddings = tf.constant([[0,0], [136, 135], [136, 135], [0,0]]), constant_values=1)
	mask = tf.constant(input_mask, dtype = tf.float32)
	
	weights = tf.get_variable('adv_weight', shape = [1, orig_image_size, orig_image_size, 3], dtype = tf.float32)
	input_image = tf.placeholder(shape = [None, new_image_size,new_image_size,1], dtype = tf.float32)
	channel_image = tf.concat([input_image, input_image, input_image], axis = -1)
	rgb_image = tf.pad(tf.concat([input_image, input_image, input_image], axis = -1), 
				paddings = tf.constant([[0,0], [int((np.ceil(orig_image_size/2.))-new_image_size/2.), int((np.floor(orig_image_size/2.))-new_image_size/2.)], [int((np.ceil(orig_image_size/2.))-new_image_size/2.), int((np.floor(orig_image_size/2.))-new_image_size/2.)], [0,0]]))

	adv_image = tf.nn.tanh(tf.multiply(weights, mask)) + rgb_image
	#print("\n shape\n")
	#print(adv_image.get_shape())

	labels = tf.placeholder(tf.float32, shape=[None, 10])
	
	from cleverhans.model import Model
	#from cleverhans.model_zoo.basic_cnn import ModelBasicCNN
	#from substitute.substitute_model_sceleton import ModelBasicCNN
	from substitute_CIFAR10 import ModelSubstitute 

	# Define TF model graph (for the black-box model)
	nb_filters = 64
	nb_classes = 10
	model = ModelSubstitute('model1', nb_classes, nb_filters)
	logits = model.get_logits(adv_image)

	# debug
	#print("\noriginal logits")
	#print(logits)
	total_logits = tf.reduce_sum(logits)
	#total_logits = tf.Print(total_logits, [total_logits], 'total_logits')
	#logits = tf.Print(logits, [logits], 'logits')

	print("Defined TensorFlow model graph.")

	#logits,_ = tf.contrib.slim.nets.inception.inception_v3(adv_image,num_classes = 1001,is_training=False)
	
	output_mapping_tensor = tf.constant(output_mapping, dtype = tf.float32)

	# debug
	#print("\noutput_mapping_tensor", output_mapping_tensor)
	#with tf.Session() as sess:
		#result = sess.run(output_mapping_tensor)
		#print(result)


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

	imported_meta = tf.train.import_meta_graph('substitute_model_CIFAR10_me.ckpt.meta')
	saver1 = tf.train.Saver([weights])
	imported_meta.restore(sess, tf.train.latest_checkpoint('.'))
	

	#ckpt = tf.train.get_checkpoint_state("./saved_model.ckpt")
	#saver.restore(sess,ckpt.model_checkpoint_path)
	print("Restored model")
'''
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)  
for i in range(total_steps) :
	batch = mnist.train.next_batch(batch_size)
	_, l, acc = sess.run([train_step, loss, accuracy], feed_dict = {input_image : preprocess(np.reshape(batch[0], [-1, 28, 28, 1])),
 									labels : batch[1]})
	if i%display_step == 0 :
		print('after %d steps the loss is %g and train_acc is %g'%(i, l, acc))
	# debug
	#print("\ntotal logits\n")
	#print(total_logits)
	#total_logits = tf.Print(total_logits, [total_logits], 'total_logits')


saver1.save(sess, './adversarial_model/adversarial_reprogramming_model.ckpt')

test_iterations = int(len(mnist.test.images)/ batch_size)
test_total_correct = 0
for i in range(test_iterations) :
 	test_imgs = preprocess(np.reshape(mnist.test.images[i*batch_size:(i+1)*batch_size], [-1,28,28,1]))
 	test_labels = mnist.test.labels[i*batch_size:(i+1)*batch_size]
 	test_corrects = sess.run(accuracy, feed_dict = {input_image : test_imgs, labels : test_labels})
 	test_total_correct += test_corrects * batch_size

print('test acc is ', float(test_total_correct)/len(mnist.test.images))

train_iterations = int(len(mnist.train.images)/ batch_size)
train_total_correct = 0
for i in range(train_iterations) :
 	train_imgs = preprocess(np.reshape(mnist.train.images[i*batch_size:(i+1)*batch_size], [-1,28,28,1]))
 	train_labels = mnist.train.labels[i*batch_size:(i+1)*batch_size]
 	train_corrects = sess.run(accuracy, feed_dict = {input_image : train_imgs, labels : train_labels})
 	train_total_correct += train_corrects * batch_size

print('train acc is ', float(train_total_correct)/len(mnist.train.images))

'''
import tensorflow_datasets as tfds
mnist_train = tfds.load(name="fashion_mnist", split=tfds.Split.TRAIN, batch_size=-1 ) 
mnist_test = tfds.load(name="fashion_mnist", split=tfds.Split.TEST, batch_size=-1)

# tfds.as_numpy return a generator that yields NumPy array records out of a tf.data.Dataset
mnist_train = tfds.as_numpy(mnist_train) 
mnist_test = tfds.as_numpy(mnist_test)

x_train, y_train = mnist_train["image"], mnist_train["label"] # seperate the x and y
x_test, y_test = mnist_test["image"], mnist_test["label"]

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

def next_batch(batch_size, data, labels):

    #Return a total of `num` random samples and labels. 

    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:batch_size]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]
    label_array = np.asarray(labels_shuffle)
    #label_array = np.expand_dims(label_array, axis=1)
    return np.asarray(data_shuffle), label_array


print("Training adversarial program")
# Train
for i in range(total_steps):
	#print("step", i, " out of ", total_steps, " total steps")
	batch_x, batch_y = next_batch(batch_size, x_train, y_train)
	#print(batch_y.shape)
	#batch_y = keras.utils.to_categorical(batch_y, 10)
	#print(batch_x.shape)
	#print(batch_y.shape)
	sess.run(train_step, feed_dict={input_image: batch_x, labels: batch_y})
print("Finished training adversarial program")


# Output accuracy
test_iterations = int(len(x_test)/ batch_size)
test_total_correct = 0
for i in range(test_iterations) :
 	test_imgs = preprocess(np.reshape(x_test[i*batch_size:(i+1)*batch_size], [-1,28,28,1]))
 	test_labels = y_test[i*batch_size:(i+1)*batch_size]
 	test_corrects = sess.run(accuracy, feed_dict = {input_image : test_imgs, labels : test_labels})
 	test_total_correct += test_corrects * batch_size

print('test acc is ', float(test_total_correct)/len(x_test))

train_iterations = int(len(x_train)/ batch_size)
train_total_correct = 0
for i in range(train_iterations) :
 	train_imgs = preprocess(np.reshape(x_train[i*batch_size:(i+1)*batch_size], [-1,28,28,1]))
 	train_labels = y_train[i*batch_size:(i+1)*batch_size]
 	train_corrects = sess.run(accuracy, feed_dict = {input_image : train_imgs, labels : train_labels})
 	train_total_correct += train_corrects * batch_size

print('train acc is ', float(train_total_correct)/len(x_train))

#train_accuracy = sess.run(accuracy, feed_dict={input_image: x_train, labels: y_train})
#test_accuracy = sess.run(accuracy, feed_dict={input_image: x_test, labels: y_test})

#print("Accuracy on train set:", train_accuracy)
#print("Accuracy on test set:", test_accuracy)



saver1.save(sess, './adversarial_model/CIFAR_MNIST.ckpt')

		        
	   

