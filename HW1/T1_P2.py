#####################
# CS 181, Spring 2022
# Homework 1, Problem 2
# Start Code
##################

# import math
# import matplotlib.cm as cm
# from math import exp
# import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from T1_P1 import kernel

# set up data
data = [(0., 0.),
        (1., 0.5),
        (2., 1),
        (3., 2),
        (4., 1),
        (6., 1.5),
        (8., 0.5)]

x_train = np.array([d[0] for d in data])
y_train = np.array([d[1] for d in data])

xy = {x: y for (x, y) in data}

x_test = np.arange(0, 12, .1)

print("y is:")
print(y_train)

def predict_knn(k=1, tau=1):
    """Returns predictions for the values in x_test, using KNN predictor with the specified k."""
    predictions = np.zeros(len(x_test))
    for i in range(len(x_test)):
        point = x_test[i]
        distances = {x: kernel(point, x, tau) for x in x_train}
        k_closest = list(sorted(distances.items(), key=lambda x: x[1], reverse=True))[0:k]
        k_closest = [x for (x, _kernel) in k_closest]
        predictions[i] = np.mean([xy[x] for x in k_closest])
    return predictions


def plot_knn_preds(k):
    plt.xlim([0, 12])
    plt.ylim([0,3])
    
    y_test = predict_knn(k=k)
    
    plt.scatter(x_train, y_train, label = "training data", color = 'black')
    plt.plot(x_test, y_test, label = "predictions using k = " + str(k))

    plt.legend()
    plt.xlabel("input coordinates, x*")
    plt.ylabel("predicted values, y=kNN(x*)")
    plt.title("KNN Predictions with k = " + str(k))
    plt.savefig('k' + str(k) + '.png')
    plt.show()

if __name__ == "__main__":
    for k in (1, 3, len(x_train)-1):
        plot_knn_preds(k)

