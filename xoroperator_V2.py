# Generate NN for XOR operation
# input layer: <NODES> nodes, one for each bit (0 = false and +1 = true)
# output layer: 1 node for result (0 = false and +1 = true)
# Use sigmoid activation function, gradient descent optimizer and mean squared error loss function
# Last update: 28.05.2019

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Define model
nodes = 2
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(nodes, input_dim=2, activation=tf.nn.sigmoid))
model.add(tf.keras.layers.Dense(1, activation=tf.nn.sigmoid))
model.compile(optimizer=tf.train.GradientDescentOptimizer(1), loss=tf.keras.losses.mean_squared_error, metrics=['binary_accuracy'])
model.summary()

# Generate train & test data
epochs = 10000
data_in = np.array([[0,0],[0,1],[1,0],[1,1]])
data_out = np.array([0,1,1,0])

# Train model
history = model.fit(data_in, data_out, epochs=epochs, verbose=0)

# Analysis of training history
for key in history.history.keys():
    plt.scatter(range(epochs), history.history[key], s=1)
    plt.ylabel(key)
    plt.xlabel('epochs')
    plt.show()

# Predict with model
result = model.predict(data_in)

# Print results
def printarray(arr):
    return np.array2string(arr).replace('\n','')

print()
print('input               ', printarray(data_in))
print('output (calculation)', printarray(data_out))
print('output (prediction) ', printarray(result))
print('output (pred. norm.)', printarray(np.round(result)))

# Get weights of model
print()
print(model.get_weights())