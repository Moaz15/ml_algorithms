import numpy as np 

class BinaryLogisticRegression:

    def __init__(self,lr= 0.01, num_iter= 1000, lambda_= 0.0, tol= 1e-4, patience=10,):
        self.lr = lr
        self.num_iter = num_iter
        self.lambda_ = lambda_
        self.theta = None 
        self.bias = None 
        self.loss_history = []
        self.tol = tol
        self.patience = patience

    @staticmethod
    def _sigmoid(z):
        # np.clip prevents numerical overflow/underflow when using exp(z)
        z = np.clip(z, -500, 500)
        return 1.0/(1.0 + np.exp(-z))
    
    def _initialize(self,n_features):
        self.theta =  np.zeros(n_features, d_type= float) 
        self.bias = 0.0

    def _compute_loss(self, y, y_hat, m):
        eps = 1e-12
        y_hat = np.clip(y_hat, eps, 1 - eps)
        loss = -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
        if self.lambda_ >0.0:
            loss += (self.lambda_ / (2 * m)) * np.sum(self.theta ** 2)
        return loss
    
    def fit_gd(self,X,y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()

        m,n = X.shape
        self._initialize(n)
        self.loss_history = []

        best_loss = float("inf")
        no_improve = 0
        
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
    


