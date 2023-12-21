import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt

data = np.loadtxt('DS0.txt')

dbscan = DBSCAN(eps=8)
clusters = dbscan.fit_predict(data)

unique_clusters = np.unique(clusters)
centers = []
for cluster in unique_clusters:
    if cluster != -1:
        cluster_points = data[clusters == cluster]
        x_avg = np.mean(cluster_points[:, 0])
        y_avg = np.mean(cluster_points[:, 1])
        centers.append((x_avg, y_avg))

vor = Voronoi(centers)

plt.figure(figsize=(9.6, 5.4), dpi=100)
plt.axis([0, 960, 0, 540])  

for center in centers:
    plt.scatter(center[0], center[1], color='blue', s=5, marker='o')

plt.scatter(data[:, 0], data[:, 1], color='black', alpha=0.1)

voronoi_plot_2d(vor, ax=plt.gca(), show_vertices=False)

plt.show()

