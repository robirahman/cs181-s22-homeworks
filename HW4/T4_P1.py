import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA = [[-3, 1], [-2, 1], [-1, -1], [0, 1], [1, -1], [2, 1], [3, 1]]
points_df = pd.DataFrame(DATA, columns=["x", "y"])


def phi(x: float) -> tuple:
    return x, (2 / 3) * x ** 4 - (8 / 3) * x ** 2


def phi_prime(x: float) -> tuple:
    return x, (7 / 12) * x ** 4 - (31 / 12) * x ** 2


if __name__ == "__main__":

    # 1.1 Plot the transformed training data and draw the maximum margin classifier.
    x1, x2 = [], []
    for x in points_df["x"]:
        _x1, _x2 = phi(x)
        x1.append(_x1)
        x2.append(_x2)
    plt.scatter(x1, x2, c=points_df["y"])
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.axhline(y=-1)
    plt.savefig("plot 1.1.png")
    plt.clf()

    # 1.2 What is the value of the margin achieved by the optimal decision boundary found in 1.1?
    # The margin is 1:
    # phi(2)[1] - phi(1)[1] = 2 is the distance between the classes
    # (phi(2)[1] + phi(1)[1])/2 = -1 is the boundary between the classes, distance 9 from the closest in each group

    # 1.3
    # v_hat = [0, 1]

    # 1.4
    #

    # 1.5
    #

    # 1.6
    #

    # 1.7
    #

    # 1.8 Using phi_prime(x), are the classes still separable?
    x1, x2 = [], []
    for x in points_df["x"]:
        _x1, _x2 = phi_prime(x)
        x1.append(_x1)
        x2.append(_x2)
    plt.scatter(x1, x2, c=points_df["y"])
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.axhline(y=-1.5)
    plt.savefig("plot 1.8.png")
    plt.clf()
    # The points are still separable, but the margin is smaller.
    # phi(2)[1] - phi(1)[1] = 1 is the new distance between the classes, so the margin is now 0.5.
