import numpy as np

# Please implement the predict() method of this class
# You can add additional private methods by beginning them with
# two underscores. It may look like the __dummyPrivateMethod below. You can feel
# free to change any of the class attributes, as long as you do not change any
# of the given function headers (they must take and return the same arguments).

class KNNModel:
    def __init__(self, k):
        self.X = None
        self.y = None
        self.n = None
        self.K = k

    def _dist(self, star1: dict, star2: dict):
        dist = 0
        dist += (star1[0] - star2[0]) ** 2
        dist += 9 * (star1[1] - star2[1]) ** 2
        return dist

    def predict(self, X_pred):
        preds = []
        for x in X_pred:
            distances = [None for _ in range(self.n)]
            for i in range(self.n):
                distances[i] = (self._dist(x, self.X[i]), self.y[i])
            distances.sort(key=lambda y: y[0])
            closest = [_[1] for _ in distances[:self.K]]
            preds.append(np.bincount(closest).argmax())
        return preds

    # In KNN, "fitting" can be as simple as storing the data, so this has been written for you
    # If you'd like to add some preprocessing here without changing the inputs, feel free,
    # but it is completely optional.
    def fit(self, X, y):
        self.X = X
        self.y = y
        self.n = len(X)
