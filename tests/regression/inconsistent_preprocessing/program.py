import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Regression fixture covering a previously failing transformed-train/transformed-test flow.
# This locks in the safe case where both partitions share the same fitted preprocessing step.

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

# Concatenating already-transformed train and test rows should not lose the "preprocessed" marking.
X_combined = np.vstack((train_transformed, test_transformed))
predictions = model.predict(X_combined)
