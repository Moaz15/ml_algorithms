import numpy as np

class MulticlassLogisticRegression:

    def __init__(self,lr=0.01, num_iter=200, tol=1e-4, patience=10,lambda_=0.0):
        self.lr= lr
        self.num_iter=  num_iter
        self.tol= tol
        self.pateince= patience
        self.lambda_= lambda_

        self.W= None   # shape(n_features,n_classes)
        self.b= None   # shape(n_classes,)
        self.loss_history= []
        self.classes_= None

    @staticmethod
    def _softmax(Z):
        # shape Z : m x k
        # numerical stabilty
        Z = Z - np.max(Z,axis=1, keepdims=True)
        expZ = np.exp(Z)
        return expZ/ np.sum(expZ, axis=1, keepdims=True)
    
    @staticmethod
    def _one_hot(y,n_classes):
        m = y.shape[0]
        Y = np.zeros(m,n_classes)
        Y[np.arange(m), y] = 1
        











        