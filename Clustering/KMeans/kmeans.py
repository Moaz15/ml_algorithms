import numpy as np
import pandas as pd 
import matplotlib as plt 

class KMeans:
    def __init__(self,K:int ,max_iters:int = 100 ,tol:float =1e-4):

        """
        Parameters : 
        K : number of clusters
        max_iters : Maximum number of iterations, by default 100
        tol : Tolerance to declare convergence, by default 1e-4

        Attributes :
        self.centroids: Stores the centroids of the clusters.
        self.labels: Stores the labels of each data point indicating which cluster it belongs to.
        self.inertia_: Stores the sum of squared distances of samples to their closest cluster center after fitting the model.
        """
        self.K = K
        self.max_iters = max_iters
        self.tol = tol
        self.centroids = None 
        self.labels = None 
        self.inertia_ = None 

    def initialize_centroids(self,X):

        """
        This method randomly shuffles the data indices and picks the first K samples as centroids.
        Simple random initialization, consider k-means++ for improvement 
        """
        indices = np.random.permutation(X.shape[0])
        centroids = X[indices[:self.K]]
        return centroids
    
    def compute_centroids(self, X, labels):
        """
        recomputes the cluster centers by averaging the points in each cluster, and handles empty clusters safely.
        """
        centroids = np.zeros((self.K , X.shape[1]))
        for k in range(self.K):
            if np.any(labels == k):
                centroids[k] = np.mean(X[labels == k],axis = 0)
            else:
                centroids[k] = X[np.random.choice(X.shape[0])]
        return centroids
    
    



    
    

        
