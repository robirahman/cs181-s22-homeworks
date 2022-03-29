# CS 181, Spring 2022
# Homework 4

from itertools import combinations, product
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Loading datasets for K-Means and HAC
small_dataset = np.load("data/small_dataset.npy")
large_dataset = np.load("data/large_dataset.npy")

CLUSTER_RESULTS = {}

def array_dist(a1, a2) -> float:
    return np.sum((a1 - a2)**2)


class KMeans(object):
    # K is the K in KMeans
    def __init__(self, K):
        self.K = K
        self.centroids = [None] * K
        self.assignments = np.zeros(1)
        self.loss = list()

    def calculate_loss(self, data):
        loss = 0
        for point in range(data.shape[0]):
            loss += array_dist(data[point], self.centroids[self.assignments[point]])
        return loss

    def update_centroids(self, data):
        for cluster in range(self.K):
            # update each cluster centroid to be at the mean of its points
            self.centroids[cluster] = np.mean(data[self.assignments == cluster], axis=0)
            # [[_[0] for _ in self.assignments == cluster]]

    def update_assignments(self, data):
        for point in range(data.shape[0]):
            distances = [array_dist(data[point], self.centroids[k]) for k in range(self.K)]
            self.assignments[point] = np.argmin(distances)

    # X is a (N x 784) array since the dimension of each image is 28x28.
    def fit(self, X):
        n, dim = X.shape
        # Forgy method (not used): randomly select K observations as initial centroids
        # self.centroids = X[np.random.choice(n, size=self.K, replace=False)]

        # Assign each image to a random cluster
        self.assignments = np.random.choice(self.K, size=n)
        # then set the centroids to the average of the images in their cluster
        self.update_centroids(X)

        for _ in range(10):  # do 10 iterations of centroid updates and cluster assignments
            # for point in range(n):  # consider each point in the dataset and assign it to the nearest cluster
            self.update_assignments(X)
            self.update_centroids(X)
            self.loss.append(self.calculate_loss(X))

        plt.clf()
        # now that there are 10 clusters, make a chart of how many images are in each cluster
        plt.title("k-means")
        plt.xlabel("cluster number")
        plt.ylabel("number of images")
        plt.hist(self.assignments)
        plt.savefig(f'plot 2.5 kmeans.png')
        plt.clf()

    # This should return the arrays for K images. Each image should represent the mean of each of the fitted clusters.
    def get_mean_images(self):
        images = np.zeros((self.K, 784))
        assignments = [None] * small_dataset.shape[0]
        for img in range(300):
            distances = [array_dist(small_dataset[img], self.centroids[k]) for k in range(self.K)]
            assignments[img] = np.argmin(distances)
        assignments = np.array(assignments)
        for cluster in range(self.K):
            images[cluster] = np.mean(small_dataset[assignments == cluster], axis=0)
        CLUSTER_RESULTS["kmeans"] = assignments
        return images


class HAC(object):
    def __init__(self, linkage):
        self.linkage = linkage
<<<<<<< Updated upstream
    
    # X is a (N x 784) array since the dimension of each image is 28x28.
=======
        self.clusters = {}

    def cluster_distance(self, cluster_1, cluster_2):
        if self.linkage == "min":
            min_dist = 2 ** 30
            for point1, point2 in product(cluster_1, cluster_2):
                new_dist = array_dist(point1, point2)
                if new_dist < min_dist:
                    min_dist = new_dist
            return min_dist  # return shortest distance between any point in cluster_1 and any point in cluster_2
        elif self.linkage == "max":
            max_dist = 0
            for point1, point2 in product(cluster_1, cluster_2):
                new_dist = array_dist(point1, point2)
                if new_dist > max_dist:
                    max_dist = new_dist
            return max_dist  # return greatest distance between any point in cluster_1 and any point in cluster_2
        elif self.linkage == "centroid":
            centroid_1 = np.mean(cluster_1, axis=0)
            centroid_2 = np.mean(cluster_2, axis=0)
            avg_dist = array_dist(centroid_1, centroid_2)
            return avg_dist  # return distance between centroid of cluster_1 and centroid of cluster_2

>>>>>>> Stashed changes
    def fit(self, X):
        num_images = X.shape[0]

        # assign each image to its own cluster
        cluster_assignments = np.arange(num_images)
        self.clusters = {c: X[cluster_assignments == c] for c in cluster_assignments}

        while len(set(cluster_assignments)) > 10:
            # compute cluster_distance between all pairs of clusters
            nearest_pair = (None, None)
            nearest_dist = 2 ** 30
            for c1, c2 in combinations(set(cluster_assignments), 2):
                dist = self.cluster_distance(self.clusters[c1], self.clusters[c2])
                if (dist < nearest_dist) and (c1 != c2):
                    nearest_dist = dist
                    nearest_pair = (c1, c2)

            # merge the two nearest clusters
            c1, c2 = nearest_pair
            cluster_assignments[cluster_assignments == c2] = c1
            self.clusters = {c: X[cluster_assignments == c] for c in cluster_assignments}

        plt.clf()
        # now that there are 10 clusters, make a chart of how many images are in each cluster
        plt.title(self.linkage)
        plt.xlabel("cluster number")
        plt.ylabel("number of images")
        plt.bar(x=list(range(10)), height=[len(_) for _ in self.clusters.values()])
        plt.savefig(f'plot 2.5 {self.linkage}.png')
        plt.clf()
        CLUSTER_RESULTS[self.linkage] = cluster_assignments

    # Returns the mean image when using n_clusters clusters
    def get_mean_images(self, n_clusters):
        return np.array([np.mean(cluster, axis=0) for cluster in self.clusters.values()])


# Plotting code for parts 2 and 3
def make_mean_image_plot(data, standardized=False):
    # Number of random restarts
    niters = 3
    K = 10
    # Will eventually store the pixel representation of all the mean images across restarts
    allmeans = np.zeros((K, niters, 784))
    for i in range(niters):
        KMeansClassifier = KMeans(K=K)
        KMeansClassifier.fit(data)
        allmeans[:, i] = KMeansClassifier.get_mean_images()
    fig = plt.figure(figsize=(10, 10))
    plt.suptitle(
        "Class mean images across random restarts"
        + (" (standardized data)" if standardized else ""),
        fontsize=16,
    )
    for k in range(K):
        for i in range(niters):
            ax = fig.add_subplot(K, niters, 1 + niters * k + i)
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_yticklabels(), visible=False)
            ax.tick_params(axis="both", which="both", length=0)
            if k == 0:
                plt.title("Iter " + str(i))
            if i == 0:
                ax.set_ylabel("Class " + str(k), rotation=90)
            plt.imshow(allmeans[k, i].reshape(28, 28), cmap="Greys_r")
    plt.show()
    if standardized:
        plt.savefig("plot 2.3.png")
    else:
        plt.savefig("plot 2.2.png")
    plt.clf()
    plt.plot(KMeansClassifier.loss)
    plt.xlabel("k-means centroid iterations")
    plt.ylabel("sum of squared distances")
    plt.title("Loss vs k-means iterations")
    plt.savefig("plot 2.1.png")


# ~~ Part 2 ~~
make_mean_image_plot(large_dataset, False)

# ~~ Part 3 ~~
means = np.mean(large_dataset, axis=0)
stdevs_nonzero = [_ or 1 for _ in np.std(large_dataset, axis=0)]
large_dataset_standardized = (large_dataset - means) / stdevs_nonzero
make_mean_image_plot(large_dataset_standardized, True)

# Plotting code for part 4
LINKAGES = ["max", "min", "centroid"]
n_clusters = 10

fig = plt.figure(figsize=(10, 10))
plt.suptitle("HAC mean images with max, min, and centroid linkages")
for l_idx, l in enumerate(LINKAGES):
    # Fit HAC
    hac = HAC(l)
    hac.fit(small_dataset)
    mean_images = hac.get_mean_images(n_clusters)
    # Make plot
    for m_idx in range(mean_images.shape[0]):
        m = mean_images[m_idx]
        ax = fig.add_subplot(
            n_clusters, len(LINKAGES), l_idx + m_idx * len(LINKAGES) + 1
        )
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.tick_params(axis="both", which="both", length=0)
        if m_idx == 0:
            plt.title(l)
        if l_idx == 0:
            ax.set_ylabel("Class " + str(m_idx), rotation=90)
        plt.imshow(m.reshape(28, 28), cmap="Greys_r")
plt.show(); plt.savefig("plot 2.4.png")

# Plotting code for part 5: see above

# TODO: Write plotting code for part 6
classifications = pd.DataFrame(CLUSTER_RESULTS, dtype=float)
for i, j in combinations(classifications.columns, 2):
    plt.clf()
    matrix = confusion_matrix(classifications[i], classifications[j])
    sns.heatmap(matrix)
    plt.savefig(f'plot 2.6 heatmap {i} vs {j}')
    plt.clf()
