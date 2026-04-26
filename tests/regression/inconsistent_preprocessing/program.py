import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# create sample data
train = np.array([[1, 2], [3, 4]])
test = np.array([[5, 6], [7, 8]])
y = np.array([0, 1])

# initialize transformer and model
scaler = StandardScaler()
model = LogisticRegression()

# transform both datasets (marked as "transformed")
train_transformed = scaler.fit_transform(train)
test_transformed = scaler.transform(test)

# fit the model
model.fit(train_transformed, y)

# predict with concatenated transformed data
X_combined = np.vstack((train_transformed, test_transformed))
predictions = model.predict(X_combined)
