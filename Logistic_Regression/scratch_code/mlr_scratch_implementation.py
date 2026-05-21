import numpy as np

class MulticlassLogisticRegression:
    def __init__(self, lr=0.01, num_iter=2000, lambda_=0.0, tol=1e-4, patience=10):
        self.lr = lr
        self.num_iter = num_iter
        self.lambda_ = lambda_
        self.tol = tol
        self.patience = patience

        self.W = None        # shape: (n_features, n_classes)
        self.b = None        # shape: (n_classes,)
        self.loss_history = []
        self.classes_ = None

    @staticmethod
    def _softmax(Z):
        # numerical stability
        Z = Z - np.max(Z, axis=1, keepdims=True)
        expZ = np.exp(Z)
        return expZ / np.sum(expZ, axis=1, keepdims=True)

    @staticmethod
    def _one_hot(y, n_classes):
        m = y.shape[0]
        Y = np.zeros((m, n_classes))
        Y[np.arange(m), y] = 1
        return Y

    def _initialize(self, n_features, n_classes):
        self.W = np.zeros((n_features, n_classes))
        self.b = np.zeros(n_classes)

    def _compute_loss(self, Y, Y_hat, m):
        eps = 1e-12
        Y_hat = np.clip(Y_hat, eps, 1 - eps)

        loss = -np.sum(Y * np.log(Y_hat)) / m
        if self.lambda_ > 0:
            loss += (self.lambda_ / (2 * m)) * np.sum(self.W ** 2)
        return loss

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).ravel()

        # encode labels to 0..K-1
        self.classes_, y_idx = np.unique(y, return_inverse=True)

        m, n = X.shape
        K = len(self.classes_)

        self._initialize(n, K)
        self.loss_history = []

        Y = self._one_hot(y_idx, K)

        best_loss = float("inf")
        no_improve = 0

        for _ in range(self.num_iter):
            # forward pass
            Z = X @ self.W + self.b
            Y_hat = self._softmax(Z)

            # loss
            loss = self._compute_loss(Y, Y_hat, m)
            self.loss_history.append(loss)

            # early stopping
            if best_loss - loss > self.tol:
                best_loss = loss
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    break

            # gradients
            error = (Y_hat - Y) / m
            dW = X.T @ error
            db = np.sum(error, axis=0)

            if self.lambda_ > 0:
                dW += (self.lambda_ / m) * self.W

            # update
            self.W -= self.lr * dW
            self.b -= self.lr * db

        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        Z = X @ self.W + self.b
        return self._softmax(Z)

    def predict(self, X):
        probs = self.predict_proba(X)
        class_idx = np.argmax(probs, axis=1)
        return self.classes_[class_idx]












        