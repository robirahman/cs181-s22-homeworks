import numpy as np
from scipy.special import softmax
import matplotlib.pyplot as plt

# Please implement the fit(), predict(), and visualize_loss() methods of this
# class. You can add additional private methods by beginning them with two
# underscores. It may look like the __dummyPrivateMethod below. You can feel
# free to change any of the class attributes, as long as you do not change any
# of the given function headers (they must take and return the same arguments).


class LogisticRegression:
    def __init__(self, eta, lam, runs=1000):
        self.eta = eta
        self.lam = lam
        self.runs = runs
        self.loss_history = []
        self.W = np.random.rand(3, 3)

    def fit(self, X, y):
        n, d = X.shape
        y = self.one_hot(y)
        X = self.bias(X)
        for _ in range(self.runs):
            for i in range(n):
                error = self.y_hat(X[i]) - y[i]
                gradient = np.matmul(error, X[i])
                update = self.eta * (gradient - self.lam * self.W) / n
                self.W -= update
            total_loss = - np.sum([np.log(self.y_hat(X[i])) * y[i] for i in range(n)])
            self.loss_history.append(total_loss)

    def one_hot(self, y):
        return np.array([[int(_ == 0), int(_ == 1), int(_ == 2)] for _ in y])

    def y_hat(self, x_star):
        logit = np.matmul(self.W, x_star)
        probabilities = softmax(logit)
        return probabilities

    def bias(self, data):
        return np.stack([np.ones(data.shape[0]), data[:, 0], data[:, 1]]).T

    def predict(self, data):
        data = self.bias(data)
        predictions = np.array([np.argmax(self.y_hat(observation)) for observation in data])
        return predictions

    def visualize_loss(self, output_file, show_charts=False):
        # in self.fit(), keep a history of the losses
        # then plot losses over iterations
        plt.figure()
        plt.plot(self.loss_history)
        plt.title("Loss vs number of gradient descent iterations")
        plt.xlabel("Number of gradient descent cycles")
        plt.ylabel("Cross-entropy loss")
        plt.show()
