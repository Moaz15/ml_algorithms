import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


from sklearn.datasets import fetch_california_housing 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import shapiro, normaltest, boxcox,spearmanr
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor


def load_data():
    california = fetch_california_housing()
    df = pd.DataFrame(california.data, columns=california.feature_names)
    df['Price'] = california.target
    return df

df = load_data()
X = df.drop('Price', axis=1)
y = df['Price']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred = lr.predict(X_test_scaled)
residuals = y_test - y_pred


def model_performance():
    print("Model Performance:")
    print(f"R² Score: {r2_score(y_test, y_pred):.3f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.3f}")


def check_linearity():
    print("\n1 Checking Linearity...")
    sns.residplot(x=y_pred, y=residuals, lowess=True, line_kws={'color': 'red'})
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title("Linearity Check: Residuals vs Predicted")
    plt.show()
    print("Random scatter ≈ linearity holds. Curved/funnel patterns → non-linearity.")

def check_independence():
    print("\n Checking Independence of Errors (Durbin-Watson)...")
    diff = np.diff(residuals)
    dw_stat = np.sum(diff**2) / np.sum(residuals**2)
    print(f"Durbin-Watson Statistic: {dw_stat:.2f}")
    if 1.5 < dw_stat < 2.5:
        print("Independence holds.")
    elif dw_stat <= 1.5:
        print("Positive autocorrelation.")
    else:
        print("Negative autocorrelation.")

def check_homoscedasticity():
    print("\nChecking Homoscedasticity...")
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.7)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title("Homoscedasticity Check: Residuals vs Predicted")
    plt.show()

    corr, p_val = spearmanr(abs(residuals), y_pred)
    print(f"Spearman correlation (|residuals| vs predictions): {corr:.3f}, p={p_val:.4f}")
    if p_val > 0.05:
        print("Homoscedasticity likely holds.")
    else:
        print("Heteroscedasticity detected — variance not constant.")

def check_normality():
    print("\nChecking Normality of Residuals...")
    sns.histplot(residuals, kde=True)
    plt.title("Histogram of Residuals (Normality Check)")
    plt.xlabel("Residuals")
    plt.show()

    stat, p1 = shapiro(residuals.sample(500, random_state=42))
    stat2, p2 = normaltest(residuals)
    print(f"Shapiro-Wilk p={p1:.4f} | D’Agostino–Pearson p={p2:.4f}")
    if p1 > 0.05 and p2 > 0.05:
        print("Residuals appear normal.")
    else:
        print("Non-normal residuals — consider log transform or outlier removal.")

def check_multicollinearity():
    print("\n Checking Multicollinearity...")
    plt.figure(figsize=(10,6))
    sns.heatmap(pd.DataFrame(X_train, columns=X_train.columns).corr(), annot=True, cmap='coolwarm', center=0)
    plt.title("Feature Correlation Matrix")
    plt.show()

    X_vif = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X_vif.columns
    vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
    print(vif_data.sort_values(by="VIF", ascending=False))
    print("\n VIF < 5 acceptable | 5–10 moderate | >10 problematic.")


def run_all():
    model_performance()
    check_linearity()
    check_independence()
    check_homoscedasticity()
    check_normality()
    check_multicollinearity()

if __name__ == '__main__':
    run_all()
