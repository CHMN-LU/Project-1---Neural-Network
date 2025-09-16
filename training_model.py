####### Try loading, training, and evaluating a model
from main import FNN
import numpy as np
# Get the data from data_loading
from data_loading import train_images, train_labels, validation_images, validation_labels, test_images, test_labels
#print(train_images.shape[0], validation_images.shape[0])

# Experiment with using binary representations of the correct integer labels
# These will be length four binary strings, therefore we use the model with 4 output nodes
# Pad each binary string up to four outputs with leading zeros
def to_binary(val: int):
    return "{:0>4b}".format(val)
def to_binary_array(val: int):
    # Take int, convert it to array where each element is one of the digits in the binary string repr
    return np.array([int(char) for char in to_binary(val)])

def vector_to_label(vec):
    return int(np.argmax(vec))
def binary_to_int(binary_str):
    return int(binary_str, 2)

# Testing binary
#vals = np.arange(10)
#binary_vals = [to_binary_array(val) for val in vals]
#print(binary_vals)

# Convert labels to binary
train_labels_binary = np.zeros((train_labels.shape[0], 4))
validation_labels_binary = np.zeros((validation_labels.shape[0], 4))
test_labels_binary = np.zeros((test_labels.shape[0], 4))
for i in range(len(train_labels)):
    train_labels_binary[i] = to_binary_array(vector_to_label(train_labels[i]))
for i in range(len(validation_labels)):
    validation_labels_binary[i] = to_binary_array(vector_to_label(validation_labels[i]))
for i in range(len(test_labels)):
    test_labels_binary[i] = to_binary_array(vector_to_label(test_labels[i]))
print(test_labels_binary.shape)

# Creates a feedforward network with 3 layers, input (28x28), hidden (30), output (4)
# The four outputs are for the binary representation of each digit
network = FNN([784, 30, 4])

# Training parameters
batch_size = 100
epochs = 10
learning_rate = 1
# Train the model
network.train_SGD(train_images, train_labels_binary, batch_size, epochs, learning_rate, validation_images, validation_labels_binary)

# Evaluate the model
print("Model after training:")
_outputs = network.evaluate(test_images, test_labels_binary, verbose=True)


