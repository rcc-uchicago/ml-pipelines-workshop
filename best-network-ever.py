
# ### A quick neural network training example - time series classification
# 
# Adapted from: https://keras.io/examples/timeseries/timeseries_classification_from_scratch/

# import packages
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# Let's do our GPU engagement check again
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
gpus = tf.config.experimental.list_physical_devices('GPU')
try:
    print(tf.config.experimental.get_device_details(gpus[0]))
except:
    print('no dice')

# Now we'll load in the data. The dataset contains 3601 training instances and another 1320 testing instances. Each timeseries corresponds to a measurement of engine noise captured by a motor sensor. For this task, the goal is to automatically detect the presence of a specific issue with the engine.
# Define a simple import function
def readucr(filename):
    data = np.loadtxt(filename, delimiter="\t")
    y = data[:, 0]
    x = data[:, 1:]
    return x, y.astype(int)

# # If on a login node (connected to internet)
# data_path = "https://raw.githubusercontent.com/hfawaz/cd-diagram/master/FordA/"

# # Else:
data_path = '/project/rcc/jdlaurence/ml-pipelines-workshop/FordA/'

# Read data, already split into test/train
x_train, y_train = readucr(data_path + "FordA_TRAIN.tsv")
x_test, y_test = readucr(data_path + "FordA_TEST.tsv")

# Next, we'll do some quick data formatting/prep

# Reformat data for network
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

# Shuffle training data
num_classes = len(np.unique(y_train))
idx = np.random.permutation(len(x_train))
x_train = x_train[idx]
y_train = y_train[idx]

# Set class -1 to be 0
y_train[y_train == -1] = 0
y_test[y_test == -1] = 0

# Define a convolutional neural network with three 1-D convolutional layers 
def make_model(input_shape):
    input_layer = keras.layers.Input(input_shape)

    conv1 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(input_layer)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.ReLU()(conv1)

    conv2 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.ReLU()(conv2)

    conv3 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.ReLU()(conv3)

    gap = keras.layers.GlobalAveragePooling1D()(conv3)

    output_layer = keras.layers.Dense(num_classes, activation="softmax")(gap)

    return keras.models.Model(inputs=input_layer, outputs=output_layer)


model = make_model(input_shape=x_train.shape[1:])

# Compile the model and train it for 50 epochs
epochs = 50 # Should be closer to 200 for performance plateau
batch_size = 32

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["sparse_categorical_accuracy"],
)
history = model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.2,
    verbose=1,
)

# %% [markdown]
# Evaluate the network

# %%
test_loss, test_acc = model.evaluate(x_test, y_test)

print("Test accuracy", test_acc)
print("Test loss", test_loss)


