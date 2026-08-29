import numpy as np

class MultipleLinearRegression:
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.beta = None  # stores coefficients (including bias)

    def _add_bias(self, X):
        """Add a column of ones for the intercept term."""
        if not self.fit_intercept:
            return X
        return np.c_[np.ones((X.shape[0], 1)), X]

    def fit_ols(self, X, y):
        """Closed-form OLS solution: β = (XᵀX)^(-1) Xᵀy"""
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)

        Xb = self._add_bias(X)
        self.beta = np.linalg.inv(Xb.T @ Xb) @ Xb.T @ y
        return self

    def fit_gd(self, X, y, lr=0.01, epochs=1000):
        """Train using batch gradient descent."""
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        Xb = self._add_bias(X)

        n, d = Xb.shape
        beta = np.zeros(d)

        for _ in range(epochs):
            y_pred = Xb @ beta
            gradient = (1/n) * Xb.T @ (y_pred - y)
            beta -= lr * gradient

        self.beta = beta
        return self

    def predict(self, X):
        Xb = self._add_bias(np.array(X, dtype=float))
        return Xb @ self.beta

    def r2_score(self, y_true, y_pred):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - ss_res / ss_tot

    def mse(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def rmse(self, y_true, y_pred):
        return np.sqrt(self.mse(y_true, y_pred))

    def mae(self, y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))
    


    

