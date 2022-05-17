from sklearn.cluster import SpectralClustering
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from numpy import random

random.seed(1)
x, _ = make_blobs(n_samples=400, centers=4, cluster_std=1.5)
plt.scatter(x[:, 0], x[:, 1])
plt.show()

sc = SpectralClustering(n_clusters=4).fit(x)
SpectralClustering(affinity='rbf', assign_labels='kmeans', coef0=1, degree=3,
                   eigen_solver=None, eigen_tol=0.0, gamma=1.0,
                   kernel_params=None, n_clusters=4, n_components=None,
                   n_init=10, n_jobs=None, n_neighbors=10, random_state=None)
labels = sc.labels_
plt.scatter(x[:, 0], x[:, 1], c=labels)
plt.show()
print(sc)
print(x)