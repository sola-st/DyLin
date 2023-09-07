from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.base import TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

import pandas as pd
import numpy as np

d = {"Inkonsistent Preprocessing Test": "InconsistentPreprocessing"}


def default_case():
    random_state = 42
    X, y = make_regression(random_state=random_state, n_features=1, noise=1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=random_state)

    scaler = StandardScaler()
    # X_train dataset is scaled but not X_test
    X_train_transformed = scaler.fit_transform(X_train)
    model = LinearRegression().fit(X_train_transformed, y_train)
    f'START;'
    model.predict(X_test)
    f'END; X_train scaled but not X_test'

    # both are scaled
    X_test_scaled = scaler.fit_transform(X_test)
    model = LinearRegression().fit(X_train_transformed, y_train)
    model.predict(X_test_scaled)


def split_afterwards():
    random_state = 42
    X, y = make_regression(random_state=random_state, n_features=1, noise=1)

    scaler = StandardScaler()
    X_transformed = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_transformed, y, test_size=0.4, random_state=random_state)

    model = LinearRegression().fit(X_train, y_train)

    # should not throw error as both train and test are transformed
    model.predict(X_test)


def access_attr():
    X = pd.DataFrame([[1.0, 2.0], [float("NaN"), 5.0], [7.0, 8.0]],
                     index=['cobra', 'viper', 'sidewinder'],
                     columns=['max_speed', 'shield'])
    X_Test = pd.DataFrame([[1.0, 2.0], [4.0, 8.0], [0.0, 8.0]],
                          index=['cobra', 'viper', 'sidewinder'],
                          columns=['max_speed', 'shield'])
    y = np.array([0, 0, 0])

    imp = SimpleImputer(missing_values=np.NAN, strategy='mean')
    imp = imp.fit(X_Test)
    X_Test = imp.transform(X_Test)
    X = imp.transform(X)
    X_Test = np.reshape(X_Test, (len(X_Test), len(X_Test[0])))
    X_Test = pd.DataFrame(X_Test)

    # returns another object which should contain marking as well -> no error
    X_Test = X_Test.loc[X_Test[0] > 0]

    LR = SVR()
    LR.fit(X, y)

    # should not throw error
    LR.predict(X_Test)


def access_attr_bad_transformation():
    X = pd.DataFrame([[1.0, 222.0], [1.0, 5.0], [7.0, 8.0]],
                     index=['cobra', 'viper', 'sidewinder'],
                     columns=['max_speed', 'shield'])
    X_Test = pd.DataFrame([[float("NaN"), 20.0], [4.0, 8.0], [0.0, 8.0]],
                          index=['cobra', 'viper', 'sidewinder'],
                          columns=['max_speed', 'shield'])
    y = np.array([0, 0, 0])

    imp = SimpleImputer(missing_values=np.NAN, strategy='mean')
    imp = imp.fit(X_Test)
    X_Test = imp.transform(X_Test)
    X_Test = np.reshape(X_Test, (len(X_Test), len(X_Test[0])))
    X_Test = pd.DataFrame(X_Test)

    LR = SVR()
    LR.fit(X, y)

    # should throw error
    f'START;'
    LR.predict(X_Test)
    f'END; LR.predict(X_Test)'


default_case()
split_afterwards()
access_attr()
access_attr_bad_transformation()
