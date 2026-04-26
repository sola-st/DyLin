import pandas as pd
import numpy as np


def pandas():
    data = {'Student ID': [10, 11, 12, 13, 14], 'Age': [23, 22, 24, 22, 25], 'Weight': [66, 72, np.inf, 68, -np.inf]}

    df = pd.DataFrame(data)  # DyLin warn

    df2 = pd.DataFrame(np.random.randn(10, 6))
    # Make a few areas have NaN values
    df2.iloc[1:3, 1] = np.inf
    # does return inf
    b = np.sum(df2)  # DyLin warn

    df3 = pd.DataFrame(np.random.randn(10, 6))
    # Make a few areas have NaN values
    df3.iloc[1:3, 1] = np.nan

    # does return finite value but contains NaN in input
    x = np.sum(df3)


def numpy():
    a = np.array([0.0, float("inf")])  # DyLin warn

    m = np.matrix([[1.0, 1.1], [float("inf"), 0.0]])  # DyLin warn

    m2 = np.matrix([[1.0, 1.1], [float("NaN"), 0.0]])

    m3 = np.matrix([[1.0, 1.1], [0.0, 0.0]])
    a2 = np.array([0, 1, 2, 3])


pandas()
numpy()
