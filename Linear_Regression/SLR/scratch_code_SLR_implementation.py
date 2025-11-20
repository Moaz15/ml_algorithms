import numpy as np

class SimpleLinearRegression:
    def __init__(self):
        self.m = 0
        self.c = 0

    def fit_ols(self,X,y):
        """
        Train the model using Ordinary Least Squares (OLS) formula.
        """
        X = np.array(X,dtype=float)
        y = np.array(y,dtype=float)

        x_mean = np.mean(X)
        y_mean = np.mean(y)

        numerator = np.sum((X-x_mean)*(y-y_mean))
        denominator = np.sum((X-x_mean)**2)

        self.m = numerator/denominator
        self.c = y_mean - self.m*x_mean

    def fit_gradient_descent(self,X,y,lr=0.01, epochs=1000):
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        n = len(X)

        m, c = 0.0, 0.0

        for _ in range(epochs):

            y_pred = m * X + c
            dm = (-2 / n) * np.sum(X * (y - y_pred))
            dc = (-2 / n) * np.sum(y - y_pred)
            m -= lr * dm
            c -= lr * dc

        self.m, self.c = m, c

    def predict(self,X):
        X = np.array(X, dtype=float)
        return self.m*X + self.c
    
    def r2_score(self,y_true,y_pred):
         y_true = np.array(y_true, dtype=float)
         y_pred = np.array(y_pred, dtype=float)

         rss = np.sum((y_true - y_pred) ** 2)
         tss = np.sum((y_true - np.mean(y_true)) ** 2)

         return 1 - (rss / tss)
    
    def mean_absolute_error(self,y_true,y_pred):
        y_true,y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs(y_true - y_pred))
    
    def mean_squared_error(self,y_true,y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean((y_pred-y_true)**2)


    
    

    
    

