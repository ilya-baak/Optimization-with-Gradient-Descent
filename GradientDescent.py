import math
import random
import numpy as np


def bgd_l2(data, y, w, eta, delta, lam, num_iter):
    new_w = w
    history_fw = []
    # Corresponds to x in the function, using the input data
    x = np.concatenate((np.full((100, 1), 1), data), axis = 1)

    for i in range(num_iter):
        w_transpose = np.transpose(new_w)
        gradient = 0
        # Note: the expressions inside the if and elif blocks correspond to the gradient
        # of the error of the linear model

        # The three cases for the gradient of the piece-wise defined function
        for j in range(len(x)):
            if (y[j] >= np.dot(w_transpose, x[j]) + delta):
                gradient += -2*(y[j] - np.dot(w_transpose, x[j]) - delta)*x[j]
            elif (abs(y[j] - np.dot(w_transpose, x[j])) < delta):
                gradient += 0
            elif (y[j] <= np.dot(w_transpose, x[j]) - delta):
                gradient += -2*(y[j] - np.dot(w_transpose, x[j]) + delta)*x[j]

        # Regularization below:
        gradient = (gradient / len(x)) + 2*lam*sum(w_transpose)
        new_w -= eta*gradient
        w_transpose = np.transpose(new_w)

        # Below is the changed objective function f
        # Again, based off the piecewise defined function (same cases as above)
        f = 0
        for j in range(len(x)):
            if (y[j] >= (np.dot(w_transpose, x[j]) + delta)):
                f += ((y[j] - np.dot(w_transpose, x[j]) - delta) ** 2)
            elif (abs(y[j] - np.dot(w_transpose, x[j])) < delta):
                f += 0
            elif (y[j] <= (np.dot(w_transpose, x[j]) - delta)):
                f += ((y[j] - np.dot(w_transpose, x[j]) + delta) ** 2)

        f = (f / len(x)) + lam*sum(w_transpose**2)
        history_fw.append(f)

    return new_w, history_fw


def sgd_l2(data, y, w, eta, delta, lam, num_iter, i=-1):
    new_w = w
    history_fw = []
    # Corresponds to x in the function, using the input data
    x = np.concatenate((np.full((100, 1), 1), data), axis=1)

    # If -1, choose i randomly, otherwise finish iteration
    if (i == -1):
        i = random.randrange(0, len(x))
    else:
        num_iter = 1

    for j in range(1, num_iter + 1):
        w_transpose = np.transpose(new_w)
        gradient = 0

        for k in range(len(x)):
            if (y[i] >= np.dot(w_transpose, x[i]) + delta):
                gradient += -2*(y[i] - np.dot(w_transpose, x[i]) - delta)*x[i]
            elif (abs(y[k] - np.dot(w_transpose, x[i])) < delta):
                gradient += 0
            elif (y[i] <= np.dot(w_transpose, x[i]) - delta):
                gradient += -2*(y[i] - np.dot(w_transpose, x[i]) + delta)*x[i]

        gradient = (gradient / len(x)) + 2*lam*sum(w_transpose)

        # Learning rate implemented below
        new_w -= (eta / math.sqrt(k))*gradient

        w_transpose = np.transpose(new_w)
        f = 0
        for k in range(len(x)):
            if (y[k] >= (np.dot(w_transpose, x[k]) + delta)):
                f += ((y[k] - np.dot(w_transpose, x[k]) - delta) ** 2)
            elif (abs(y[k] - np.dot(w_transpose, x[k])) < delta):
                f += 0
            elif (y[k] <= (np.dot(w_transpose, x[k]) - delta)):
                f += ((y[k] - np.dot(w_transpose, x[k]) + delta) ** 2)

        f = (f / len(x)) + lam*sum(w_transpose**2)
        history_fw.append(f)
        i = random.randrange(0, len(x))

    return new_w, history_fw
