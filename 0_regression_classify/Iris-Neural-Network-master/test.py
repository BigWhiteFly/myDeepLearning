# Dependencies
import numpy as np
import pandas as pd
import tensorflow as tf

# Make results reproducible
seed = 1234

np.random.seed(seed)
tf.set_random_seed(seed)

# Loading the dataset
dataset = pd.read_csv('0302_DeepLearning\\code\\0_regression_classify\\Iris-Neural-Network-master\\Iris_Dataset.csv')
dataset = pd.get_dummies(dataset, columns=['Species']) # One Hot Encoding
values = list(dataset.columns.values)

y = dataset[values[-3:]]
y = np.array(y, dtype='float32')
X = dataset[values[1:-3]]
X = np.array(X, dtype='float32')

indices = np.random.choice(len(X), len(X), replace=False)
X_values = X[indices]
y_values = y[indices]
# Session
sess = tf.Session()

# Interval / Epochs
interval = 50
epoch = 500

# Initialize placeholders
X_data = tf.placeholder(shape=[None, 4], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 3], dtype=tf.float32)

# Input neurons : 4
# Hidden neurons : 8
# Output neurons : 3
hidden_layer_nodes = 8

# Create variables for Neural Network layers
w1 = tf.Variable(tf.random_normal(shape=[4,hidden_layer_nodes])) # Inputs -> Hidden Layer
b1 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes]))   # First Bias
print(b1)
