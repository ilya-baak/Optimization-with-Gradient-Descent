import random
import numpy as np
import matplotlib.pyplot as plt
from Problem5 import bgd_l2, sgd_l2

if __name__ == '__main__':
    data = np.load('data.npy')
    splitArr = np.hsplit(data, 2)
    y = splitArr[1]
    x = splitArr[0]
    w = np.random.random(2)

# Gradient Descent below

#Parameters .05, .1, .001, 50
    new_w, history_fw = bgd_l2(x, y, w, .05, .1, .001, 50)
    plt.plot(history_fw)
    plt.title("Gradient Descent with Parameters .05, .1, .001, 50")
    plt.xlabel("Number of Iterations")
    plt.ylabel("The Objective Function")
    plt.show()

# Parameters .1, .01, .001, 50
    new_w, history_fw = bgd_l2(x, y, w, .1, .01, .001, 50)
    plt.plot(history_fw)
    plt.title("Gradient Descent with Parameters .1, .01, .001, 50")
    plt.xlabel("Number of Iterations")
    plt.ylabel("The Objective Function")
    plt.show()
# Parameters .1, 0, .001, 100
    new_w, history_fw = bgd_l2(x, y, w, .1, 0, .001, 100)
    plt.plot(history_fw)
    plt.title("Gradient Descent with Parameters .1, 0, .001, 100")
    plt.xlabel("Number of Iterations")
    plt.ylabel("The Objective Function")
    plt.show()
# Parameters .1, 0, 0, 100
    new_w, history_fw = bgd_l2(x, y, w, .1, 0, 0, 100)
    plt.plot(history_fw)
    plt.title("Gradient Descent with Parameters .1, 0, 0, 100")
    plt.xlabel("Number of Iterations")
    plt.ylabel("The Objective Function")
    plt.show()


# Stochastic Gradient Descent Below

# Parameters 1, 0.1, 0.5, 800
    new_w, history_fw = sgd_l2(x, y, w, 1, 0.1, 0.5, 800)
    plt.plot(history_fw)
    plt.title("Stochastic Gradient Descent with Parameters 1, 0.1, 0.5, 800")
    plt.xlabel("Number of Iterations")
    plt.ylabel("The Objective Function")
    plt.show()
# Parameters 1, 0.01, 0.1, 800
    new_w, history_fw = sgd_l2(x, y, w, 1, 0.01, 0.1, 800)
    plt.plot(history_fw)
    plt.title("Stochastic Gradient Descent with Parameters 1, 0.01, 0.1, 800")
    plt.xlabel("Number of Iterations")
    plt.ylabel("The Objective Function")
    plt.show()
# Parameters 1, 0, 0, 40
    new_w, history_fw = sgd_l2(x, y, w, 1, 0, 0, 40)
    plt.plot(history_fw)
    plt.title("Stochastic Gradient Descent with Parameters 1, 0, 0, 40")
    plt.xlabel("Number of Iterations")
    plt.ylabel("The Objective Function")
    plt.show()
# Parameters 1, 0, 0, 8000
    new_w, history_fw = sgd_l2(x, y, w, 1, 0, 0, 800)
    plt.plot(history_fw)
    plt.title("Stochastic Gradient Descent with Parameters 1, 0, 0, 8000")
    plt.xlabel("Number of Iterations")
    plt.ylabel("The Objective Function")
    plt.show()