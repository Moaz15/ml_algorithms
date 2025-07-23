import numpy as np 

class LogisticRegressionScratch:
    def __init__(self,lr = 0.01, num_iter=100 , lambda_ = 0.0):
        self.lr = lr
        self.num_iter = num_iter
        self.lambda_ = lambda_
        self.theta = None 
        self.bias = None 
        self.loss_history = []
    
    def sigmoid(self,z):
        return 1/(1+np.exp(-z))
    
    def initialize(self,n_features):
        self.theta =  np.zeros(n_features) 
        self.bias = 0

    def fit(self,X,y):
        m,n = X.shape
        self.initialize(n)

        for i in range(self.num_iter):
            z = np.dot(X,self.theta) + self.bias
            y_hat = self.sigmoid(z)

            error = y_hat - y

            # L2-regularized gradients
            d_theta = (1/m) * np.dot(X.T, error) + (self.lambda_/m) * self.theta
            d_bias = (1/m) * np.sum(error)

            # Update parameters
            self.theta -= self.lr * d_theta
            self.bias -= self.lr * d_bias

            # regularized loss
            loss = -np.mean(y * np.log(y_hat + 1e-9) + (1 - y) * np.log(1 - y_hat + 1e-9))
            loss += (self.lambda_ / (2*m)) * np.sum(self.theta ** 2)

            if i%100 == 0:
                print(f"Iteration {i}, Loss: {loss:.4f}")
            self.loss_history.append(loss)
    
    def predict_prob(self,X):
        z = np.dot(X,self.theta) + self.bias
        return self.sigmoid(z)
    
    def predict(self,X,threshold = 0.5):
        return(self.predict_prob(X) >=threshold).astype(int)
    





        





