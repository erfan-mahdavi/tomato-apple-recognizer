import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
from dnn_app_utils_v3 import *



import matplotlib.pyplot as plt

# Set default size of plots
plt.rcParams['figure.figsize'] = (5.0, 4.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# To enable interactive mode (useful if you're plotting in VSCode)
plt.ion()

# If you need to reload modules manually (since %autoreload won't work):
import importlib

# Example of how to reload a module (if needed)
# import your_module
# importlib.reload(your_module)

# Your plotting or other code would go here


np.random.seed(1)

import os
import numpy as np
import tensorflow as tf

def load_and_process_image(image_path, target_size=(64, 64)):
    """
    Load and preprocess an image: resize, convert to RGB if grayscale, and flatten.
    """
    # Load the image
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image)
    
    # Convert grayscale to RGB if needed
    if image.shape[-1] == 1:
        image = tf.image.grayscale_to_rgb(image)

    # Resize the image
    image = tf.image.resize(image, target_size)

    # Normalize the image to [0, 1]
    image = image / 255.0

    # Flatten the image to a column vector
    image_flattened = tf.reshape(image, [-1])  # Shape (height * width * 3,)

    return image_flattened.numpy()

# Define paths to the directories
apple_dir = "C:\\play\\programming\\python\\machine_learning\\apple_tomato\\test\\apples"
tomato_dir = "C:\\play\\programming\\python\\machine_learning\\apple_tomato\\test\\tomatoes"

# Initialize lists to hold the data and labels
x_test = []
y_test = []

# Load and process apple images
for img_name in os.listdir(apple_dir):
    img_path = os.path.join(apple_dir, img_name)  # Full path to the image
    image_vector = load_and_process_image(img_path)
    x_test.append(image_vector)
    y_test.append(1)  # Label for apple is 1

# Load and process tomato images
for img_name in os.listdir(tomato_dir):
    img_path = os.path.join(tomato_dir, img_name)  # Full path to the image
    image_vector = load_and_process_image(img_path)
    x_test.append(image_vector)
    y_test.append(0)  # Label for tomato is 0

# Convert the lists to NumPy arrays
x_test = np.array(x_test).T  # Transpose to get images as columns
y_test = np.array(y_test).reshape(1, -1)  # Reshape to be a row vector

# Print the shapes of X and Y
print(f"X_test shape: {x_test.shape}")  # Should be (height*width*channels, number_of_images)
print(f"Y_test shape: {y_test.shape}")  # Should be (1, number_of_images)

apple_dir = "C:\\play\\programming\\python\\machine_learning\\apple_tomato\\train\\apples"
tomato_dir = "C:\\play\\programming\\python\\machine_learning\\apple_tomato\\train\\tomatoes"

# Initialize lists to hold the data and labels
X = []
Y = []

# Load and process apple images
for img_name in os.listdir(apple_dir):
    img_path = os.path.join(apple_dir, img_name)  # Full path to the image
    image_vector = load_and_process_image(img_path)
    X.append(image_vector)
    Y.append(1)  # Label for apple is 1

# Load and process tomato images
for img_name in os.listdir(tomato_dir):
    img_path = os.path.join(tomato_dir, img_name)  # Full path to the image
    image_vector = load_and_process_image(img_path)
    X.append(image_vector)
    Y.append(0)  # Label for tomato is 0

# Convert the lists to NumPy arrays
X = np.array(X).T  # Transpose to get images as columns
Y = np.array(Y).reshape(1, -1)  # Reshape to be a row vector

# Print the shapes of X and Y
print(f"X shape: {X.shape}")  # Should be (height*width*channels, number_of_images)
print(f"Y shape: {Y.shape}")  # Should be (1, number_of_images)






def two_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):
    """
    Implements a two-layer neural network: LINEAR->RELU->LINEAR->SIGMOID.
    
    Arguments:
    X -- input data, of shape (n_x, number of examples)
    Y -- true "label" vector (containing 1 if cat, 0 if non-cat), of shape (1, number of examples)
    layers_dims -- dimensions of the layers (n_x, n_h, n_y)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- If set to True, this will print the cost every 100 iterations 
    
    Returns:
    parameters -- a dictionary containing W1, W2, b1, and b2
    """
    
    np.random.seed(1)
    grads = {}
    costs = []                              # to keep track of the cost
    m = X.shape[1]                           # number of examples
    (n_x, n_h, n_y) = layers_dims
    
    # Initialize parameters dictionary, by calling one of the functions you'd previously implemented
    ### START CODE HERE ### (≈ 1 line of code)
    parameters = initialize_parameters(n_x, n_h, n_y)
    ### END CODE HERE ###
    
    # Get W1, b1, W2 and b2 from the dictionary parameters.
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    # Loop (gradient descent)

    for i in range(0, num_iterations):

        # Forward propagation: LINEAR -> RELU -> LINEAR -> SIGMOID. Inputs: "X, W1, b1, W2, b2". Output: "A1, cache1, A2, cache2".
        ### START CODE HERE ### (≈ 2 lines of code)
        A1, cache1 = linear_activation_forward(X, W1, b1, activation="relu")
        A2, cache2 = linear_activation_forward(A1, W2, b2, activation="sigmoid")
        ### END CODE HERE ###
        
        # Compute cost
        ### START CODE HERE ### (≈ 1 line of code)
        cost = compute_cost(A2, Y)
        ### END CODE HERE ###
        
        # Initializing backward propagation
        dA2 = - (np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))
        
        # Backward propagation. Inputs: "dA2, cache2, cache1". Outputs: "dA1, dW2, db2; also dA0 (not used), dW1, db1".
        ### START CODE HERE ### (≈ 2 lines of code)
        dA1, dW2, db2 = linear_activation_backward(dA2, cache2, activation="sigmoid")
        dA0, dW1, db1 = linear_activation_backward(dA1, cache1, activation="relu")
        ### END CODE HERE ###
        
        # Set grads['dWl'] to dW1, grads['db1'] to db1, grads['dW2'] to dW2, grads['db2'] to db2
        grads['dW1'] = dW1
        grads['db1'] = db1
        grads['dW2'] = dW2
        grads['db2'] = db2
        
        # Update parameters.
        ### START CODE HERE ### (approx. 1 line of code)
        parameters = update_parameters(parameters, grads, learning_rate)
        ### END CODE HERE ###

        # Retrieve W1, b1, W2, b2 from parameters
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]
        
        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        if print_cost and i % 100 == 0:
            costs.append(cost)
       
    # plot the cost

    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters


### CONSTANTS DEFINING THE MODEL ####
n_x = 12288     # num_px * num_px * 3
n_h2 = 7
n_h3 = 5
n_h4 = 3
n_y = 1
layers_dims = (n_x,n_h2,n_h3,n_h4, n_y)
def L_layer_model(X, Y, layers_dims, learning_rate = 0.00315, num_iterations = 3000, print_cost=False):#lr was 0.009
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.
    
    Arguments:
    X -- data, numpy array of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    np.random.seed(1)
    costs = []                         # keep track of cost
    
    # Parameters initialization. (≈ 1 line of code)
    ### START CODE HERE ###
    parameters = initialize_parameters_deep(layers_dims)
    ### END CODE HERE ###
    
    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        ### START CODE HERE ### (≈ 1 line of code)
        AL, caches = L_model_forward(X, parameters)
        ### END CODE HERE ###
        
        # Compute cost.
        ### START CODE HERE ### (≈ 1 line of code)
        cost = compute_cost(AL, Y)
        ### END CODE HERE ###
    
        # Backward propagation.
        ### START CODE HERE ### (≈ 1 line of code)
        grads = L_model_backward(AL, Y, caches)
        ### END CODE HERE ###
 
        # Update parameters.
        ### START CODE HERE ### (≈ 1 line of code)
        parameters = update_parameters(parameters, grads, learning_rate)
        ### END CODE HERE ###
                
        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
            
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters


parameters = L_layer_model(X, Y, layers_dims, num_iterations = 3000, print_cost = True)
pred_train = predict(X, Y, parameters)
pred_test = predict(x_test, y_test, parameters)
print("training data accurecy : " + str(pred_train))
print("test data accurecy : " + str(pred_test))
