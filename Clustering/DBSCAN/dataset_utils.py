from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler

def create_dataset(n_samples =500, noise =0.5 , random_state=42 ):
    X, _ = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    return X

def scale_features(X):
    scaler = StandardScaler()
    scaled_X = scaler.fit_transform(X)
    return scaled_X 


