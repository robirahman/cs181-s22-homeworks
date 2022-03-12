#####################
# CS 181, Spring 2022
# Homework 1, Problem 1
# STARTER CODE
##################

import numpy as np
import matplotlib.pyplot as plt

data = [(0., 0.),
        (1., 0.5),
        (2., 1.),
        (3., 2.),
        (4., 1.),
        (6., 1.5),
        (8., 0.5)]


def compute_loss(tau):
    loss = 0.0
    for i in range(len(data)):
        loss += (data[i][1] - kernel_predict(data, i, tau)) ** 2
    return loss


def kernel_predict(data: list, observation: int, t: float) -> float:
    yhat = 0
    for i in range(len(data)):
        yhat += 0 if i == observation else kernel(data[observation][0], data[i][0], t) * data[i][1]
    return yhat


def kernel(x1: float, x2: float, tau: float) -> float:
    return np.exp(-((x1-x2) ** 2) / tau)


for tau in (0.01, 2, 100):
    print("Loss for tau = " + str(tau) + ": " + str(compute_loss(tau)))


def f_xstar(data: list, xstar: float, tau: float) -> float:
    yhat = 0
    for i in range(len(data)):
        yhat += kernel(data[i][0], xstar, tau) * data[i][1]
    return yhat


if __name__ == "__main__":
    x_stars = np.arange(0, 12, 0.1)
    yhats_point01 = [f_xstar(data, x_star, 0.01) for x_star in x_stars]
    yhats_2 = [f_xstar(data, x_star, 2) for x_star in x_stars]
    yhats_100 = [f_xstar(data, x_star, 100) for x_star in x_stars]

    plt.plot([0,1,2,3,4,6,8],[0,0.5,1,2,1,1.5,0.5], alpha=0.7, label="y (actual)")
    plt.plot(x_stars, yhats_point01, alpha=0.3, label="tau=0.01")
    plt.plot(x_stars, yhats_2, alpha=0.3, label="tau=2")
    plt.plot(x_stars, yhats_100, alpha=0.3, label="tau=100")
    plt.xlabel("input coordinates, x*")
    plt.ylabel("predicted values, y=f(x*)")
    plt.title("Predicted y values vs kernel lengthscale")
    plt.legend()
    plt.show()
