from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from pandas import DataFrame
import seaborn as sns

d = {"Data leakage test": "ObjectMarkingAnalysis",
     "configName": "leaked_data"}

# defines dataset
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0,
	n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=1)

# oversampling datasets , new rows are synthesized based on existing rows 
X_new ,y_new = SMOTE().fit_resample(X,y)
# splits after over -sampling no longer produce independent train/test data 
f'START;'
X_train , X_test , y_train , y_test = train_test_split( X_new , y_new , test_size=0.2, random_state =42)
f'END;'

rf = RandomForestClassifier().fit(X_train ,y_train) 
rf.predict(X_test)