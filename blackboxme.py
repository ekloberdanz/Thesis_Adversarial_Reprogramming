import pandas as pd
from tensorflow.python.keras import backend as K
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
#from keras import Sequential
#from tensorflow.keras.models import Sequential
#from keras.layers.convolutional import Conv2D, MaxPooling2D
#from keras.layers import Dense, Flatten

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D

from sklearn.metrics import confusion_matrix
import time
from keras.optimizers import *
from matplotlib.pyplot import *
import matplotlib.pyplot as plt
import numpy
import inspect
from tabulate import tabulate
import sys


# Assume access to data, but not to model => train substitute

#####################################################################################################################
# CREATE A SUBSTITUTE MODEL
#####################################################################################################################

# Import MNIST dataset from keras API
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# Reshape image data 
image_size = 28 * 28 # each train and test example is 28 by 28 pixels
X_train = X_train.reshape(X_train.shape[0], image_size)
X_test = X_test.reshape(X_test.shape[0], image_size)

# Encode training labels to 1-of-c output
y_train = keras.utils.to_categorical(y_train,num_classes=10)
y_test = keras.utils.to_categorical(y_test,num_classes=10)


# Function for 1 layer neural net
def createNN_1layer(filters, kernel_size, strides, padding, output_dim, activation, loss, weights, bias, learning_rate, optimizer_type):
	classifier = Sequential()
	#First Hidden Layer
	classifier.add(Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, activation=activation, kernel_initializer=weights, bias_initializer=bias, input_shape=(28,28,1)))
	classifier.add(MaxPooling2D(2, 2))
	classifier.add(Flatten())
	#Output Layer (10 digits => 10 neurons)
	classifier.add(Dense(output_dim, activation='softmax', kernel_initializer=weights, bias_initializer=bias))
	#Compiling the neural network
	classifier.compile(optimizer = 'Adam',loss=loss, metrics =['accuracy'])
	return(classifier, filters, kernel_size, strides, padding, output_dim, activation, loss, weights, bias, learning_rate, optimizer_type)


results = []	
classifier2 = createNN_1layer(filters=64, kernel_size = (5, 5), strides=(1,1), padding='valid', output_dim=10, activation='relu', loss='categorical_crossentropy', weights='random_normal', bias='zeros',	learning_rate = 0.001, optimizer_type=Adam)[0]
# model2
model2 = createNN_1layer(filters=64, kernel_size = (5, 5), strides=(1,1), padding='valid', output_dim=10, activation='relu', loss='categorical_crossentropy', weights='random_normal', bias='zeros', learning_rate = 0.001, optimizer_type=Adam)[1:]
# Start time of training
start_time2=time.time()
# Trainng
batch_size2 = 10
epochs2 = 10
# Set up Early Stopping of training
callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss',min_delta=0,patience=2,verbose=0,mode='auto')]

history2 = classifier2.fit(X_train,y_train, validation_split=.2, batch_size=batch_size2, epochs=epochs2,  callbacks=callbacks)
# Execution time
execution_time2 = time.time() - start_time2
# Evaluate (train data): losss and accuracy
eval_model2=classifier2.evaluate(X_train, y_train)
x=model2
eval_model2_test=classifier2.evaluate(X_test, y_test)
results.append([1, batch_size2, epochs2, x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9],'Adam',x[11], execution_time2, eval_model2[0], eval_model2[1], eval_model2_test[0], eval_model2_test[1]])


substitute_model = classifier2

###############################################################################################################################################
# TRAIN AN ADVERSARIAL PROGRAM ON SUBSTITUE MODEL
###############################################################################################################################################

# Masking matrix
print("Preparing Masking Matrix")
M = np.ones((299,299,3)).astype('float32')
M[135:163,135:163,:] = 0

# Adverserial Reprogramming layer
class MyLayer(Layer):
    def __init__(self, W_regularizer=0.05, **kwargs):
        self.init = initializers.get('glorot_uniform')
        self.W_regularizer = regularizers.l2(W_regularizer)
        super(MyLayer, self).__init__(**kwargs)
    def build(self, input_shape):
        assert len(input_shape) == 4
        # Create a trainable weight variable for this layer.
        self.W = self.add_weight(name='kernel', 
                                      shape=(299,299,3),
                                      initializer=self.init, regularizer=self.W_regularizer,
                                      trainable=True)
        super(MyLayer, self).build(input_shape)  # Be sure to call this at the end
    def call(self, x):
        prog = K.tanh(self.W*M)
        out = x + prog
        return out
    def compute_output_shape(self, input_shape):
        return (input_shape[0],input_shape[1],input_shape[2],input_shape[3])

# Create Adversarial Input
x = Input(shape=input_shape)
x_aug = ZeroPadding2D(padding=((135,136),(135,136)))(x)
out = MyLayer()(x_aug)
probs = substitute_model(out)

model = Model(inputs=x,outputs=probs)

# Freezing InceptionV3 model
model.layers[-1].trainable = False

print(model.summary())

adam = Adam(lr=0.05,decay=0.48)
model.compile(loss='categorical_crossentropy', optimizer = adam, metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1,
          validation_data=(X_train[:100],y_train[:100]))


score = model.evaluate(X_train, y_train, verbose=0)
print('Train loss:', score[0])
print('Train accuracy:', score[1])

score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

model.save_weights('adversarial.h5')

data = np.concatenate((X_train,X_test),axis=0)
label = np.concatenate((y_train,y_test),axis=0)

pred = model.predict(data)

top_probs = np.zeros((10))

top_probs_idx = np.zeros((10)).astype('int')

for i in range(len(data)):
     if np.argmax(label[i])==np.argmax(pred[i]) and pred[i][np.argmax(pred[i])]>=top_probs[np.argmax(pred[i])]:
             top_probs[np.argmax(pred[i])] = pred[i][np.argmax(pred[i])]
             top_probs_idx[np.argmax(pred[i])] = i

interim_model = Model(inputs=model.input,outputs=model.layers[-2].output)

imgs = np.zeros((10,299,299,3)).astype('float32')

for i, ind in enumerate(top_probs_idx):
	prog = interim_model.predict(data[ind:ind+1])
	imgs[i:i+1] = prog

np.save("imgs.npy")

## Saving them as png files

for i in range(10):
	fig = np.around((imgs[i] + 1.0) / 2.0 * 255)
	fig = fig.astype(np.uint8).squeeze()
	pic = Image.fromarray(fig)
	pic.save("%d_new.png"%i)