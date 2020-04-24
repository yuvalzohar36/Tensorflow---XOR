'''
Importing Tensorflow, numpy
define the training data, target data
create the neural network model.
input_dim (2 dimensional array == 2).
activation function == relu/sigmoid
Dense layer - using for 2 dimmesional array input
model.compile ---> specify the compiler
fit the model (train it)
predict the results

'''
import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense

training_data = np.array([[0,0],[0,1],[1,0],[1,1]], "float32")
target_data = np.array([[0],[1],[1],[0]], "float32")

model = Sequential()
model.add(Dense(64, input_dim=2, activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['binary_accuracy'])
model.fit(training_data, target_data, nb_epoch=500, verbose=2)
print (model.predict(training_data))
