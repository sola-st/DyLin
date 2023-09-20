import tensorflow as tf
from tensorflow.keras import Model, layers
import numpy as np
import tensorflow_datasets as tfds
import random

from tensorflow.keras.datasets import mnist
tf.get_logger().setLevel('INFO')

d = {"Analyse Tensorflow Gradients Test": "GradientAnalysis"}

# Hyperparameters
num_classes = 10  # total classes (0-9 digits).
num_features = 784  # data features (img shape: 28*28).

# Training Parameters
learning_rate = 0.001
training_steps = 100
batch_size = 5
display_step = 100

# Network Parameters
# MNIST image shape is 28*28px, we will then handle 28 sequences of 28 timesteps for every sample.
num_input = 28  # number of sequences.
timesteps = 28  # timesteps.
num_units = 5  # number of neurons for the LSTM layer.

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Convert to float32.
x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)
# Flatten images to 1-D vector of 784 features (28*28).
x_train, x_test = x_train.reshape(
    [-1, 28, 28]), x_test.reshape([-1, num_features])
# Normalize images value from [0, 255] to [0, 1].
x_train, x_test = x_train / 255., x_test / 255.

# Use tf.data API to shuffle and batch data.
train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data = train_data.repeat().shuffle(5000).batch(batch_size).prefetch(1)

# Create LSTM Model.


class Net(Model):
    # Set layers.
    def __init__(self):
        super(Net, self).__init__()
        # RNN (LSTM) hidden layer.
        self.lstm_layer = layers.LSTM(units=num_units)
        self.out = layers.Dense(num_classes)

    # Set forward pass.
    def __call__(self, x, is_training=False):
        # LSTM layer.
        x = self.lstm_layer(x)
        # Output layer (num_classes).
        x = self.out(x)
        if not is_training:
            # tf cross entropy expect logits without softmax, so only
            # apply softmax when not training.
            x = tf.nn.softmax(x)
        return x


# Build LSTM model.
network = Net()

# Cross-Entropy Loss.
# Note that this will apply 'softmax' to the logits.


def cross_entropy_loss(x, y):
    # Convert labels to int 64 for tf cross-entropy function.
    y = tf.cast(y, tf.int64)
    # Apply softmax to logits and compute cross-entropy.
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=x)
    # Average loss across the batch.
    return tf.reduce_mean(loss)

# Accuracy metric.


def accuracy(y_pred, y_true):
    # Predicted class is the index of highest score in prediction vector (i.e. argmax).
    correct_prediction = tf.equal(
        tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis=-1)


# Adam optimizer.
optimizer = tf.optimizers.Adam(learning_rate)

# Optimization process.


def run_optimization(x, y):
    # Wrap computation inside a GradientTape for automatic differentiation.
    with tf.GradientTape() as tape:
        # Forward pass.
        pred = network(x, is_training=True)
        # Compute loss.
        loss = cross_entropy_loss(pred, y)

    # Variables to update, i.e. trainable variables.
    trainable_variables = network.trainable_variables

    # Compute gradients.
    gradients = tape.gradient(loss, trainable_variables)

    """
    Insert value above threshold at random tensor from gradient list
    at random index
    """
    def add_value_at_random_location(tensor: tf.Tensor):
        random_indices = []
        for i in tensor.shape:
            random_indices.append(random.randint(0,i-1))
        value = float(20.1)
        if random.randint(0,1) == 1:
            value = -value

        indices = [random_indices]
        delta = tf.SparseTensor(indices=indices, values=[value], dense_shape=list(tensor.shape))
        res = tensor + tf.sparse.to_dense(delta)
        return res
    tensor_index = random.randint(0,len(gradients)-1)
    rand_tensor = gradients[tensor_index]
    rand_tensor = add_value_at_random_location(rand_tensor)
    gradients[tensor_index] = rand_tensor

    # Clip-by-value would be best practice here to prevent the issue 
    #gradients = [(tf.clip_by_value(grad, clip_value_min=-1.0, clip_value_max=1.0))
                 #for grad in gradients]
    # Update weights following gradients.
    f'START;'
    optimizer.apply_gradients(zip(gradients, trainable_variables))
    f'END; bad tensor at index {tensor_index}'


# Run training for the given number of steps.
for step, (batch_x, batch_y) in enumerate(train_data.take(training_steps), 1):
    # Run the optimization to update W and b values.
    run_optimization(batch_x, batch_y)

    if step % display_step == 0:
        pred = network(batch_x, is_training=True)
        loss = cross_entropy_loss(pred, batch_y)
        acc = accuracy(pred, batch_y)

        # run['monitoring/logs/loss'].log(loss)
        # run['monitoring/logs/acc'].log(acc)

        print("step: %i, loss: %f, accuracy: %f" % (step, loss, acc))
