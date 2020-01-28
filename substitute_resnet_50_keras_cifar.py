"""
This tutorial shows how to generate adversarial examples
using FGSM in black-box setting.
The original paper can be found at:
https://arxiv.org/abs/1602.02697
"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import functools
import logging
import numpy as np
from six.moves import xrange
import tensorflow as tf

from cleverhans.attacks import FastGradientMethod
from cleverhans.utils_tf import jacobian_graph, jacobian_augmentation
from cleverhans.compat import flags
from cleverhans.initializers import HeReLuNormalInitializer
from cleverhans.loss import CrossEntropy
from cleverhans.model import Model
from cleverhans.train import train
from cleverhans.utils import set_log_level
from cleverhans.utils import TemporaryLogLevel
from cleverhans.utils import to_categorical
from cleverhans.utils_tf import model_eval
from cleverhans.evaluation import batch_eval

from cleverhans.model import Model
import os
import keras
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
import random
#from keras.applications import resnet50
from substitute_model_sceleton import ModelBasicCNN
#from tensorflow.contrib.slim.nets 
from nets import resnet_v2
import tensorflow.contrib.slim as slim
from keras import Sequential, layers
from keras.layers import Dense
#Custom
from load_images import load_images


#from nets import resnet_utils
from nets.resnet_utils import *
from keras.models import load_model
from sklearn.datasets import load_files   
from keras.utils import np_utils
from keras import applications
from keras.preprocessing.image import ImageDataGenerator 
from keras import optimizers
from keras.models import Sequential, Model as Model_Keras,load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D,GlobalAveragePooling2D
from keras.callbacks import TensorBoard,ReduceLROnPlateau,ModelCheckpoint
from keras.layers import Dense, Flatten
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import SGD, Adam

FLAGS = flags.FLAGS
NB_CLASSES = 100
BATCH_SIZE = 128
LEARNING_RATE = .001
NB_EPOCHS = 10
HOLDOUT = 5000
DATA_AUG = 6
NB_EPOCHS_S = 10
LMBDA = .1
AUG_BATCH_SIZE = 512



def train_oracle(x_train, y_train, x_test, y_test, img_rows, img_cols, nchannels, nb_classes):



  model = Sequential()
  model.add(Conv2D(16, (3, 3), activation='relu',  input_shape=(img_rows,img_cols,nchannels)))
  model.add(Conv2D(32, (3, 3), padding ='same', activation = 'relu',input_shape= (img_rows, img_cols, nchannels)))
  model.add(MaxPooling2D(2, 2))
  model.add(Flatten())
  model.add(Dense(nb_classes, activation='softmax', kernel_initializer='random_normal'))  #Output Layer
  
  '''
  model = Sequential()
  model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
  model.add(MaxPooling2D((2, 2)))
  model.add(Conv2D(64, (3, 3), activation='relu'))
  model.add(MaxPooling2D((2, 2)))
  model.add(Conv2D(64, (3, 3), activation='relu'))
  model.add(Flatten())
  model.add(Dense(64, activation='relu'))
  model.add(Dense(100, activation='softmax'))

  #x_train = x_train[:1000]
  #y_train = y_train[:1000]
  base_model = applications.resnet50.ResNet50(weights= None, include_top=False, input_shape= (32,32,3))

  x = base_model.output
  x = GlobalAveragePooling2D()(x)
  x = Dropout(0.5)(x)
  predictions = Dense(100, activation= 'softmax')(x)
  model = Model_Keras(inputs = base_model.input, outputs = predictions)
  

  '''

  adam = Adam(lr=0.001)
  model.compile(optimizer= adam, loss='categorical_crossentropy', metrics=['accuracy'])
  model.fit(x_train, y_train, validation_split=0.33, epochs = 1, batch_size = 50, shuffle=True)

  train_accuracy = model.evaluate(x_train, y_train)
  print ("Loss = " + str(train_accuracy[0]))
  print ("Train Accuracy = " + str(train_accuracy[1]))

  test_accuracy = model.evaluate(x_test, y_test)
  print ("Loss = " + str(test_accuracy[0]))
  print ("Test Accuracy = " + str(test_accuracy[1]))

  model.summary()
  return model


class ModelSubstitute(Model):
  def __init__(self, scope, nb_classes, nb_filters=200, **kwargs):
    del kwargs
    Model.__init__(self, scope, nb_classes, locals())
    self.nb_filters = nb_filters

  def fprop(self, x, **kwargs):
    del kwargs
    my_dense = functools.partial(
        tf.layers.dense, kernel_initializer=HeReLuNormalInitializer)
    with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
      #y = tf.layers.flatten(x)
      y = tf.layers.Conv2D(filters = 64, kernel_size = (5, 5), activation='relu')(x)
      y = tf.layers.MaxPooling2D(pool_size=(2, 2), strides =(1, 1), padding='same')(y)
      y = tf.layers.flatten(y)
      y = my_dense(y, self.nb_filters, activation=tf.nn.relu)
      y = my_dense(y, self.nb_filters, activation=tf.nn.relu)
      logits = my_dense(y, self.nb_classes)
      return {self.O_LOGITS: logits,
              self.O_PROBS: tf.nn.softmax(logits=logits)}


def train_sub(sess, x, y, black_box_model, x_sub, y_sub, nb_classes,
              nb_epochs_s, batch_size, learning_rate, data_aug, lmbda,
              aug_batch_size, rng, img_rows, img_cols,
              nchannels):
  """
  This function creates the substitute by alternatively
  augmenting the training data and training the substitute.
  :param sess: TF session
  :param x: input TF placeholder
  :param y: output TF placeholder
  :param bbox_preds: output of black-box model predictions
  :param x_sub: initial substitute training data
  :param y_sub: initial substitute training labels
  :param nb_classes: number of output classes
  :param nb_epochs_s: number of epochs to train substitute model
  :param batch_size: size of training batches
  :param learning_rate: learning rate for training
  :param data_aug: number of times substitute training data is augmented
  :param lmbda: lambda from arxiv.org/abs/1602.02697
  :param rng: numpy.random.RandomState instance
  :return:
  """

  # Define TF model graph (for the black-box model)
  model_sub = ModelSubstitute('model_s', nb_classes)
  preds_sub = model_sub.get_logits(x)
  loss_sub = CrossEntropy(model_sub, smoothing=0)

  print("Defined TensorFlow model graph for the substitute.")

  # Define the Jacobian symbolically using TensorFlow
  grads = jacobian_graph(preds_sub, x, nb_classes)

  # Train the substitute and augment dataset alternatively
  for rho in xrange(data_aug):
    print("Substitute training epoch #" + str(rho))
    train_params = {
        'nb_epochs': nb_epochs_s,
        'batch_size': batch_size,
        'learning_rate': learning_rate
    }
    with TemporaryLogLevel(logging.WARNING, "cleverhans.utils.tf"):
      train(sess, loss_sub, x_sub, to_categorical(y_sub, nb_classes),
            init_all=False, args=train_params, rng=rng,
            var_list=model_sub.get_params())

    # If we are not at last substitute training iteration, augment dataset
    if rho < data_aug - 1:
      print("Augmenting substitute training data.")
      # Perform the Jacobian augmentation
      lmbda_coef = 2 * int(int(rho / 3) != 0) - 1
      x_sub = jacobian_augmentation(sess, x, x_sub, y_sub, grads, lmbda_coef * lmbda, aug_batch_size)

      print("Labeling substitute training data.")
      # Label the newly generated synthetic points by quering the oracle
      y_sub = np.hstack([y_sub, y_sub])
      x_sub_prev = x_sub[int(len(x_sub)/2):]
      bbox_preds = black_box_model.predict(x_sub_prev, batch_size=batch_size)
      bbox_labels = bbox_preds.argmax(axis=-1)
      print(bbox_labels)
      y_sub[int(len(x_sub)/2):] = bbox_labels # the oracle only gives labels, not probabilities
  return model_sub, preds_sub


def cifar100_blackbox(train_start=0, train_end=60000, test_start=0,
                   test_end=10000, nb_classes=NB_CLASSES,
                   batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE,
                   nb_epochs=NB_EPOCHS, holdout=HOLDOUT, data_aug=DATA_AUG,
                   nb_epochs_s=NB_EPOCHS_S, lmbda=LMBDA,
                   aug_batch_size=AUG_BATCH_SIZE):
  """
  arxiv.org/abs/1602.02697
  :param train_start: index of first training set example
  :param train_end: index of last training set example
  :param test_start: index of first test set example
  :param test_end: index of last test set example
  :return: a dictionary with:
           * black-box model accuracy on test set
           * substitute model accuracy on test set
           * black-box model accuracy on adversarial examples transferred
             from the substitute model
  """

  # Set logging level to see debug information
  set_log_level(logging.DEBUG)

  # Create TF session
  config = tf.ConfigProto()
  config.gpu_options.per_process_gpu_memory_fraction = 0.4
  sess = tf.Session(config=config)

  (x_train,y_train), (x_test,y_test) = keras.datasets.cifar100.load_data()

  print('x_train shape:', x_train.shape)
  print('x_test shape:', x_test.shape)
  print(x_train.shape[0], 'train samples')
  print(x_test.shape[0], 'test samples')

  # input image dimensions
  img_rows, img_cols, nchannels = x_train.shape[1], x_train.shape[2], x_train.shape[3]

  x_train = x_train.astype('float32')
  x_test = x_test.astype('float32')
  
  # Normalize
  x_train /= 255.
  x_test /= 255.

  # convert class vectors to binary class matrices
  y_train = keras.utils.to_categorical(y_train, nb_classes)
  y_test = keras.utils.to_categorical(y_test, nb_classes)

  print ("number of training examples = " + str(x_train.shape[0]))
  print ("number of test examples = " + str(x_test.shape[0]))
  print ("X_train shape: " + str(x_train.shape))
  print ("Y_train shape: " + str(y_train.shape))
  print ("X_test shape: " + str(x_test.shape))
  print ("Y_test shape: " + str(y_test.shape))

  # Initialize substitute training set reserved for training substitute model
  x_sub = x_test[:holdout]
  y_sub = np.argmax(y_test[:holdout], axis=1)

  # Redefine test set as remaining samples available only for training oracle
  x_test = x_test[holdout:]
  y_test = y_test[holdout:]

  # Define input TF placeholders
  x = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols, nchannels))              
  y = tf.placeholder(tf.float32, shape=(None, nb_classes))

  # Seed random number generator for reproducible results
  rng = np.random.RandomState([2017, 8, 30])

  # Simulate the oracle model locally
  print("Training the oracle")
  black_box_model = train_oracle(x_train, y_train, x_test, y_test, img_rows, img_cols, nchannels, nb_classes)
  print("Oracle ready")

  # Prepare for saving trained substitute model
  saver = tf.train.Saver()

  # Train substitute using method from https://arxiv.org/abs/1602.02697
  print("Training the substitute model.")
  train_sub_out = train_sub(sess, x, y, black_box_model, x_sub, y_sub,
                            nb_classes, nb_epochs_s, batch_size,
                            learning_rate, data_aug, lmbda, aug_batch_size,
                            rng, img_rows, img_cols, nchannels)
  model_sub, preds_sub = train_sub_out
  
  print("Finished training substitute")

  # Evaluate the substitute model
  eval_params = {'batch_size': batch_size}

  train_acc = model_eval(sess, x, y, preds_sub, x_train, y_train, args=eval_params)
  print('Train accuracy of substitute model: ' + str(train_acc))

  test_acc = model_eval(sess, x, y, preds_sub, x_test, y_test, args=eval_params)
  print('Test accuracy of substitute model: ' + str(test_acc))

  # Save model after it is trained
  save_path = saver.save(sess, "./substitute_model_RESNET50.ckpt")
  print("Model saved in path: %s" % save_path)


def main(argv=None):
  cifar100_blackbox(nb_classes=FLAGS.nb_classes, batch_size=FLAGS.batch_size,
                 learning_rate=FLAGS.learning_rate,
                 nb_epochs=FLAGS.nb_epochs, holdout=FLAGS.holdout,
                 data_aug=FLAGS.data_aug, nb_epochs_s=FLAGS.nb_epochs_s,
                 lmbda=FLAGS.lmbda, aug_batch_size=FLAGS.data_aug_batch_size)


if __name__ == '__main__':

  # General flags
  flags.DEFINE_integer('nb_classes', NB_CLASSES,
                       'Number of classes in problem')
  flags.DEFINE_integer('batch_size', BATCH_SIZE,
                       'Size of training batches')
  flags.DEFINE_float('learning_rate', LEARNING_RATE,
                     'Learning rate for training')

  # Flags related to oracle
  flags.DEFINE_integer('nb_epochs', NB_EPOCHS,
                       'Number of epochs to train model')

  # Flags related to substitute
  flags.DEFINE_integer('holdout', HOLDOUT,
                       'Test set holdout for adversary')
  flags.DEFINE_integer('data_aug', DATA_AUG,
                       'Number of substitute data augmentations')
  flags.DEFINE_integer('nb_epochs_s', NB_EPOCHS_S,
                       'Training epochs for substitute')
  flags.DEFINE_float('lmbda', LMBDA, 'Lambda from arxiv.org/abs/1602.02697')
  flags.DEFINE_integer('data_aug_batch_size', AUG_BATCH_SIZE,
                       'Batch size for augmentation')

  tf.app.run()