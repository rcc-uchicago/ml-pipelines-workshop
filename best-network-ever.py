# import packages
import pandas as pd
import numpy as np
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split

# Check for GPU engagement
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
gpus = tf.config.experimental.list_physical_devices('GPU')
print(tf.config.experimental.get_device_details(gpus[0]))

# Download some data about mushrooms
# #If on a login-node (with internet):
# url = 'https://raw.githubusercontent.com/rcc-uchicago/ml-pipelines-workshop/main/mushrooms.csv'
url = '/project/rcc/jdlaurence/ml-pipelines-workshop/mushrooms.csv'
data = pd.read_csv(url)

# Prepare data for network
data2 = pd.get_dummies(data) # One-hot encoding of categorical/string variables

# Train-test-split
X = data2.drop(['class_e', 'class_p'], axis=1)
y = data2[['class_e', 'class_p']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, shuffle=True)

# define model
model = Sequential()
model.add(Dense(units=16, activation='sigmoid', input_shape=[X_train.shape[1]])) # 16 neuron layer
model.add(Dense(2, activation='softmax')) # Binary output

model.compile(loss='binary_crossentropy', optimizer='adam', metrics='accuracy')

# Train model
model.fit(X_train, y_train, epochs=25, verbose=0)

# Evaluate
loss_and_acc = model.evaluate(X_test,y_test)
print('Loss: ',loss_and_acc[0])
print('Accuracy: ', loss_and_acc[1]