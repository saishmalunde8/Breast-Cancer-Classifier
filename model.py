import time
import numpy as np
import matplotlib.pyplot as plt
from dnn_utils_v2 import *
import tkinter as tk

costs = []
# ---------------------------------------------------------------------------------------------------------------------------

def __compute_accuracy(AL, Y):
    AL = np.where(AL >= 0.5, 1, 0)

    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0

    for i in range(AL.shape[1]):
        if(AL[0, i] == 1):
            if (Y[0, i] == AL[0, i]):
                true_positives += 1
            else:
                false_positives += 1
        else:
            if (Y[0, i] == AL[0, i]):
                true_negatives += 1
            else:
                false_negatives += 1

    accuracy = ((true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)) * 100

    return accuracy

# Public functions -------------------------------------------------------------------------------------------------------------
def plot_cost_iteration():
    if (costs != []):
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per hundreds)')
        plt.title("Learning rate")
        plt.show()
    else:
        print("Costs are empty")

def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):#lr was 0.009
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.
    
    Arguments:
    X -- data, numpy array of shape (no. of features, number of examples)
    Y -- true "label" vector of shape (1, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    np.random.seed(1)
    global costs                        # keep track of cost
    
    # Parameters initialization. (≈ 1 line of code)
    parameters = initialize_parameters_deep(layers_dims)
    
    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        AL, caches = L_model_forward(X, parameters)
        
        # Compute cost.
        cost = compute_cost(AL, Y)
    
        # Backward propagation.
        grads = L_model_backward(AL, Y, caches)
 
        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)
                
        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
            
    # # plot the cost
    # plt.plot(np.squeeze(costs))
    # plt.ylabel('cost')
    # plt.xlabel('iterations (per hundreds)')
    # plt.title("Learning rate =" + str(learning_rate))
    # plt.show()

    print("Model has been trained.")
    
    return parameters

def evaluate(X, Y, parameters):
    np.random.seed(1)
    costs = []                         # keep track of cost

    # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
    AL, caches = L_model_forward(X, parameters)

    # Compute cost.
    cost = compute_cost(AL, Y)

    accuracy = __compute_accuracy(AL, Y)

    return accuracy
    
    # print(f"Cost: {cost}")
    # print(f"Accuracy: {accuracy}")

def predict(X, parameters):
    np.random.seed(1)
    costs = []                         # keep track of cost

    # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
    AL, caches = L_model_forward(X, parameters)

    AL = np.where(AL >= 0.5, 1, 0)

    prediction = "Malignant" if (AL == 1) else "Benign"

    tk.messagebox.showinfo("Prediction", f"Patient's Cancer is classified as {prediction}.")
    
    return AL

print("model.py compiled successfully!")