import pandas as pd
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names).head(1000)
y = pd.Series(housing.target, name='Price').head(500)
X_y = pd.concat([X, y], axis=1)

print(X_y.head(3))
X_y.to_csv('housing.csv')