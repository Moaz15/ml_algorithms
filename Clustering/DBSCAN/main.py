from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler 
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt 
import numpy as np

# Create non-linear dataset
X, _ = make_moons(n_samples=500, noise=0.05, random_state=42)

# Standarize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

dbscan = DBSCAN(eps=0.3, min_samples=5)
dbscan.fit(X_scaled)

labels = dbscan.labels_

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

n_noise = np.sum(labels == -1)
print(f"Number of noise points : {n_noise}")

print("Cluster labels:", set(labels))
# print(f"Number of clusters found: {n_clusters}")


# plt.figure(figsize=(8,6))
# plt.scatter(X_scaled[:,0],X_scaled[:, 1],c=labels, cmap='plasma', s=30)
# plt.title("DBSCAN Clustering (eps=0.3, min_samples=5)")
# plt.xlabel("Feature 1")
# plt.ylabel("Feature 2")
# plt.show()