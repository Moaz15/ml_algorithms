import numpy as np 
import matplotlib.pyplot as plt
import joblib

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss

from scratch_implementation import LogisticRegressionScratch 

# synthetic ctr like dataset 
X, y = make_classification(n_samples=1000, n_features=5, 
                           n_informative=3, n_redundant=1,
                           random_state=42)
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)


# Define hyperparameter grid 
learning_rates = [0.01, 0.05, 0.1]
num_iters = [500, 1000]
thresholds = [0.4, 0.5, 0.6]
lambdas = [0.0, 0.01, 0.1, 1.0]

best_f1 = 0
best_params = {}

for lr in learning_rates:
    for n_iter in num_iters:
        for lam in lambdas:
            model = LogisticRegressionScratch(lr=lr, num_iter=n_iter, lambda_=lam)
            model.fit(X_train, y_train)
            probs = model.predict_prob(X_test)
            for thresh in thresholds:
                preds = (probs >= thresh).astype(int)
                f1 = f1_score(y_test, preds)
                print(f"lr={lr}, iter={n_iter}, λ={lam}, threshold={thresh} → F1: {f1:.4f}")
                if f1 > best_f1:
                    best_f1 = f1
                    best_params = {
                        'lr': lr,
                        'num_iter': n_iter,
                        'lambda_': lam,
                        'threshold': thresh
                    }
print("\n✅ Best Hyperparameters:")
print(best_params)
print(f"Best F1 Score: {best_f1:.4f}")


best_model = LogisticRegressionScratch(
    lr=best_params['lr'],
    num_iter=best_params['num_iter'],
    lambda_=best_params['lambda_']
)

best_model.fit(X_train, y_train)
probs = best_model.predict_prob(X_test)
preds = (probs >= best_params['threshold']).astype(int)

print("\n📊 Final Evaluation on Best Model:")
print("Accuracy :", accuracy_score(y_test, preds))
print("Precision:", precision_score(y_test, preds))
print("Recall   :", recall_score(y_test, preds))
print("F1 Score :", f1_score(y_test, preds))
print("Log Loss :", log_loss(y_test, probs))


# plt.figure(figsize=(8, 5))
# plt.plot(best_model.loss_history, label="Training Loss", color="blue")
# plt.title("Loss vs Iterations")
# plt.xlabel("Iteration")
# plt.ylabel("Loss")
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()


np.save("theta.npy", best_model.theta)
np.save("bias.npy", best_model.bias)

joblib.dump(scaler, "scaler.pkl")