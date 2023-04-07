# Optimization-with-Gradient-Descent
This Python module implements two types of gradient descent algorithms, Batch Gradient Descent (BGD) and Stochastic Gradient Descent (SGD), with L2 regularization. The algorithms are used for solving a linear regression problem with a piece-wise defined objective function. The module includes functions for performing BGD and SGD with L2 regularization, as well as plotting the objective function during the optimization process.

## Functions
The module contains two main functions:

bgd_l2(data, y, w, eta, delta, lam, num_iter): Performs Batch Gradient Descent with L2 regularization on the input data. It takes the following parameters as input:

data: Input data, a 2D numpy array of shape (m, n), where m is the number of samples and n is the number of features.
y: Target values, a 1D numpy array of shape (m,).
w: Initial weights, a 1D numpy array of shape (n+1,).
eta: Learning rate, a float value.
delta: Threshold for the piece-wise defined objective function, a float value.
lam: Regularization parameter, a float value.
num_iter: Number of iterations, an integer value.

It returns the updated weights and a list containing the values of the objective function during the optimization process.

sgd_l2(data, y, w, eta, delta, lam, num_iter, i=-1): Performs Stochastic Gradient Descent with L2 regularization on the input data. It takes the same parameters as bgd_l2, with an additional optional parameter i which specifies the index of the sample to start the optimization from. If i is set to -1, a random sample is chosen as the starting point. It returns the updated weights and a list containing the values of the objective function during the optimization process.
The module also includes a plotting function:

plot_objective_function(history_fw): Plots the values of the objective function during the optimization process. It takes a list of values of the objective function as input and creates a plot showing the change in the objective function over the iterations.
