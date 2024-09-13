#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import plotly.express as px
from pylab import rcParams
import warnings
from sklearn.preprocessing import StandardScaler
import seaborn as sns 
rcParams["figure.figsize"]=(30,18)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.size'] = 15
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))  


# # **Data Overview and Cleaning**

# In[ ]:


df= pd.read_csv("/kaggle/input/titanic/train.csv")


# In[ ]:


df


# In[ ]:


Col = df.columns
Col


# In[ ]:


df.infer_objects


# In[ ]:


missing_values = df.isnull().sum()
missing_values


# In[ ]:


# 1. Fill missing Age values with the median
df['Age'].fillna(df['Age'].median(), inplace=True)

# 2. Fill missing Cabin values with 'Unknown'
df['Cabin'].fillna('Unknown', inplace=True)

# 3. Fill missing Embarked values with the mode
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# 4. Verify that there are no more missing values
missing_values = df.isnull().sum()


# # **Univariate Analysis**

# In[ ]:


fig, axes = plt.subplots(2, 2, figsize=(14, 10))

sns.countplot(x='Survived', data=df, palette='Set2', ax=axes[0, 0])
axes[0, 0].set_title('Survival Distribution')
axes[0, 0].set_xlabel('Survived (0 = No, 1 = Yes)')
axes[0, 0].set_ylabel('Count')

sns.countplot(x='Pclass', data=df, palette='Set2', ax=axes[0, 1])
axes[0, 1].set_title('Passenger Class Distribution')
axes[0, 1].set_xlabel('Passenger Class')
axes[0, 1].set_ylabel('Count')

sns.countplot(x='Sex', data=df, palette='Set2', ax=axes[1, 0])
axes[1, 0].set_title('Gender Distribution')
axes[1, 0].set_xlabel('Gender')
axes[1, 0].set_ylabel('Count')

sns.countplot(x='Embarked', data=df, palette='Set2', ax=axes[1, 1])
axes[1, 1].set_title('Port of Embarkation Distribution')
axes[1, 1].set_xlabel('Embarked')
axes[1, 1].set_ylabel('Count')

plt.tight_layout()
plt.show()


# In[ ]:


plt.figure(figsize=(8, 6))
sns.histplot(df['Age'], bins=30, kde=True, color='blue')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()


# In[ ]:


plt.figure(figsize=(8, 6))
sns.histplot(df['Fare'], bins=30, kde=True, color='green')
plt.title('Fare Distribution')
plt.xlabel('Fare')
plt.ylabel('Frequency')
plt.show()


# In[ ]:


fig, axes = plt.subplots(2, 2, figsize=(14, 10))

sns.countplot(x='Survived', data=df, palette='coolwarm', ax=axes[0, 0])
axes[0, 0].set_title('Survival Distribution')
for p in axes[0, 0].patches:
    percentage = '{:.1f}%'.format(100 * p.get_height() / len(df))
    axes[0, 0].annotate(percentage, (p.get_x() + 0.3, p.get_height() + 10), ha='center')

sns.countplot(x='Pclass', hue='Survived', data=df, palette='Set2', ax=axes[0, 1])
axes[0, 1].set_title('Passenger Class vs. Survival')
for p in axes[0, 1].patches:
    percentage = '{:.1f}%'.format(100 * p.get_height() / len(df))
    axes[0, 1].annotate(percentage, (p.get_x() + 0.3, p.get_height() + 10), ha='center')

sns.countplot(x='Sex', hue='Survived', data=df, palette='Set1', ax=axes[1, 0])
axes[1, 0].set_title('Gender vs. Survival')
for p in axes[1, 0].patches:
    percentage = '{:.1f}%'.format(100 * p.get_height() / len(df))
    axes[1, 0].annotate(percentage, (p.get_x() + 0.3, p.get_height() + 10), ha='center')

sns.countplot(x='Embarked', hue='Survived', data=df, palette='Set3', ax=axes[1, 1])
axes[1, 1].set_title('Embarkation Port vs. Survival')
for p in axes[1, 1].patches:
    percentage = '{:.1f}%'.format(100 * p.get_height() / len(df))
    axes[1, 1].annotate(percentage, (p.get_x() + 0.3, p.get_height() + 10), ha='center')

plt.tight_layout()
plt.show()


# In[ ]:


fig, axes = plt.subplots(3, 2, figsize=(14, 12))

sns.histplot(df['Age'], bins=30, kde=True, color='blue', ax=axes[0, 0])
axes[0, 0].set_title('Age Distribution (Histogram + KDE)')
axes[0, 0].set_xlabel('Age')
axes[0, 0].set_ylabel('Density')

sns.histplot(df['Fare'], bins=30, kde=True, color='green', ax=axes[0, 1])
axes[0, 1].set_title('Fare Distribution (Histogram + KDE)')
axes[0, 1].set_xlabel('Fare')
axes[0, 1].set_ylabel('Density')

sns.violinplot(x='Survived', y='Age', data=df, palette='Set2', ax=axes[1, 0])
axes[1, 0].set_title('Age Distribution by Survival (Violin Plot)')
axes[1, 0].set_xlabel('Survived')
axes[1, 0].set_ylabel('Age')

sns.violinplot(x='Pclass', y='Fare', data=df, palette='Set2', ax=axes[1, 1])
axes[1, 1].set_title('Fare Distribution by Pclass (Violin Plot)')
axes[1, 1].set_xlabel('Pclass')
axes[1, 1].set_ylabel('Fare')

sns.boxplot(x='SibSp', data=df, palette='cool', ax=axes[2, 0])
axes[2, 0].set_title('Sibling/Spouse Distribution (Boxplot)')
axes[2, 0].set_xlabel('SibSp')

sns.boxplot(x='Parch', data=df, palette='cool', ax=axes[2, 1])
axes[2, 1].set_title('Parent/Children Distribution (Boxplot)')
axes[2, 1].set_xlabel('Parch')

plt.tight_layout()
plt.show()


# In[ ]:


fig, axes = plt.subplots(2, 2, figsize=(14, 10))

sns.stripplot(x='Pclass', y='Age', data=df, palette='Set2', ax=axes[0, 0], jitter=True)
axes[0, 0].set_title('Age vs. Passenger Class (Strip Plot)')
axes[0, 0].set_xlabel('Pclass')
axes[0, 0].set_ylabel('Age')

sns.swarmplot(x='Pclass', y='Fare', data=df, palette='Set2', ax=axes[0, 1])
axes[0, 1].set_title('Fare vs. Passenger Class (Swarm Plot)')
axes[0, 1].set_xlabel('Pclass')
axes[0, 1].set_ylabel('Fare')

sns.violinplot(x='Sex', y='Age', data=df, palette='Set3', ax=axes[1, 0])
axes[1, 0].set_title('Age Distribution by Gender (Violin Plot)')
axes[1, 0].set_xlabel('Sex')
axes[1, 0].set_ylabel('Age')

sns.stripplot(x='Survived', y='Age', data=df, palette='cool', ax=axes[1, 1], jitter=True)
axes[1, 1].set_title('Age vs. Survival (Strip Plot)')
axes[1, 1].set_xlabel('Survived')
axes[1, 1].set_ylabel('Age')

plt.tight_layout()
plt.show()


# # ** Bivariate Analysis**

# In[ ]:


fig, axes = plt.subplots(1, 3, figsize=(18, 6))

sns.barplot(x='Pclass', y='Survived', data=df, ax=axes[0], palette='Set2')
axes[0].set_title('Survival Rate by Passenger Class')
axes[0].set_ylabel('Survival Rate')

sns.barplot(x='Sex', y='Survived', data=df, ax=axes[1], palette='Set1')
axes[1].set_title('Survival Rate by Gender')
axes[1].set_ylabel('Survival Rate')

sns.barplot(x='Embarked', y='Survived', data=df, ax=axes[2], palette='Set3')
axes[2].set_title('Survival Rate by Embarkation Port')
axes[2].set_ylabel('Survival Rate')

plt.tight_layout()
plt.show()


# In[ ]:


numerical_cols = ['Age', 'Fare', 'SibSp', 'Parch', 'Survived']
sns.pairplot(df[numerical_cols], hue='Survived', diag_kind='kde', palette='coolwarm')
plt.suptitle('Pairplot for Numerical Features with Survival Hue', y=1.02)
plt.show()


# In[ ]:


fig, axes = plt.subplots(2, 2, figsize=(14, 10))

sns.boxplot(x='Survived', y='Age', data=df, ax=axes[0, 0], palette='Set2')
axes[0, 0].set_title('Age Distribution by Survival')

sns.boxplot(x='Survived', y='Fare', data=df, ax=axes[0, 1], palette='Set1')
axes[0, 1].set_title('Fare Distribution by Survival')

sns.boxplot(x='Survived', y='SibSp', data=df, ax=axes[1, 0], palette='Set3')
axes[1, 0].set_title('SibSp Distribution by Survival')

sns.boxplot(x='Survived', y='Parch', data=df, ax=axes[1, 1], palette='cool')
axes[1, 1].set_title('Parch Distribution by Survival')

plt.tight_layout()
plt.show()


# In[ ]:


plt.figure(figsize=(10, 6))
sns.stripplot(x='Embarked', y='Fare', hue='Survived', data=df, palette='coolwarm', jitter=True, dodge=True)
plt.title('Fare Distribution by Embarkation Port and Survival')
plt.show()


# In[ ]:


from scipy.stats import chi2_contingency

contingency_table = pd.crosstab(df['Pclass'], df['Survived'])
chi2, p, dof, expected = chi2_contingency(contingency_table)

contingency_table = pd.crosstab(df['Sex'], df['Survived'])
chi2, p, dof, expected = chi2_contingency(contingency_table)

contingency_table = pd.crosstab(df['Embarked'], df['Survived'])
chi2, p, dof, expected = chi2_contingency(contingency_table)


# Chi-Square Test: Used to determine if there's a statistically significant relationship between two categorical variables

# In[ ]:


df['Fare_log'] = np.log1p(df['Fare'])
plt.figure(figsize=(10, 6))
sns.histplot(df['Fare_log'], bins=30, kde=True, color='green')
plt.title('Log-Transformed Fare Distribution (Histogram + KDE)')
plt.show()


# # **Multivariate Analysis**

# In[ ]:


g = sns.FacetGrid(df, col='Pclass', row='Sex', hue='Survived', height=4, aspect=1.5)
g.map(sns.scatterplot, 'Age', 'Fare', alpha=0.7)
g.add_legend()
g.set_axis_labels('Age', 'Fare')
plt.subplots_adjust(top=0.9)
g.fig.suptitle('Survival Rate by Pclass, Sex, and Embarked', fontsize=16)
plt.show()


# In[ ]:


from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(df['Fare'], df['Age'], df['Survived'], c=df['Survived'], cmap='coolwarm', alpha=0.8)
ax.set_xlabel('Fare')
ax.set_ylabel('Age')
ax.set_zlabel('Survived')
ax.set_title('3D Plot of Fare, Age, and Survival')
plt.colorbar(scatter)
plt.show()


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Select relevant features and preprocess
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# Features and target variable
X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
y = df['Survived']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluation


# # ** Feature Engineering**

# In[ ]:


df['FamilySize'] = df['SibSp'] + df['Parch'] + 1  # +1 to include the passenger


# In[ ]:


df['IsAlone'] = (df['FamilySize'] == 1).astype(int)


# In[ ]:


df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

# Group rare titles into 'Other'
df['Title'] = df['Title'].replace(['Lady', 'Countess', 'Capt', 'Col',
                                    'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Other')
df['Title'] = df['Title'].replace('Mlle', 'Miss')
df['Title'] = df['Title'].replace('Ms', 'Miss')
df['Title'] = df['Title'].replace('Mme', 'Mrs')


# In[ ]:


df['AgeBin'] = pd.cut(df['Age'], bins=[0, 12, 18, 35, 60, 100], labels=['Child', 'Teen', 'Adult', 'Middle Age', 'Senior'])


# In[ ]:


df['Age'] = df.groupby(['Pclass', 'Sex'])['Age'].transform(lambda x: x.fillna(x.median()))


# In[ ]:


df['HasCabin'] = df['Cabin'].notnull().astype(int)


# In[ ]:


df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)


# In[ ]:


# Encoding categorical variables: Sex, Embarked, Title, AgeBin
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
df = pd.get_dummies(df, columns=['Title', 'AgeBin'], drop_first=True)  # One-hot encoding


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Train the model
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

# Predict on test set
y_pred_logreg = log_reg.predict(X_test)

# Evaluate the model


# In[ ]:


from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)

y_pred_rf = rf_clf.predict(X_test)



# In[ ]:


from xgboost import XGBClassifier

xgb_clf = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
xgb_clf.fit(X_train, y_train)

# Predict on test set
y_pred_xgb = xgb_clf.predict(X_test)



# In[ ]:


from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_rf = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=param_grid, cv=5, verbose=2, n_jobs=-1)
grid_rf.fit(X_train, y_train)


y_pred_rf_tuned = grid_rf.best_estimator_.predict(X_test)


# In[ ]:


from sklearn.metrics import roc_auc_score

roc_auc_logreg = roc_auc_score(y_test, log_reg.predict_proba(X_test)[:, 1])

roc_auc_rf = roc_auc_score(y_test, rf_clf.predict_proba(X_test)[:, 1])

roc_auc_xgb = roc_auc_score(y_test, xgb_clf.predict_proba(X_test)[:, 1])



# # **Ensemble Modeling**

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)

y_pred_rf = rf_clf.predict(X_test)


# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier

gb_clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_clf.fit(X_train, y_train)

y_pred_gb = gb_clf.predict(X_test)


# In[ ]:


from xgboost import XGBClassifier

xgb_clf = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
xgb_clf.fit(X_train, y_train)

y_pred_xgb = xgb_clf.predict(X_test)


# In[ ]:


from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

base_models = [
    ('log_reg', LogisticRegression(max_iter=1000)),
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'))
]

meta_model = LogisticRegression(max_iter=1000)

stacking_clf = StackingClassifier(estimators=base_models, final_estimator=meta_model)
stacking_clf.fit(X_train, y_train)

y_pred_stack = stacking_clf.predict(X_test)


# In[ ]:


from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)

rfe = RFE(model, n_features_to_select=10)
rfe = rfe.fit(X_train, y_train)

selected_features = X_train.columns[rfe.support_]


# In[ ]:


importances = rf_clf.feature_importances_
indices = importances.argsort()[::-1]

for f in range(X_train.shape[1]):
    print(f"{f + 1}. feature {X_train.columns[indices[f]]} ({importances[indices[f]]:.4f})")


# In[ ]:


from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(rf_clf, X, y, cv=5, scoring='accuracy')


# In[ ]:


from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X, y)



# In[ ]:


from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

fpr, tpr, _ = roc_curve(y_test, xgb_clf.predict_proba(X_test)[:, 1])
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


# In[ ]:


test_data = pd.read_csv('/kaggle/input/titanic/test.csv') 


# In[ ]:


import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def preprocess_data(data):
    numeric_features = ['Age', 'Fare']  
    categorical_features = ['Embarked', 'Sex']  
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])
    
    processed_data = preprocessor.fit_transform(data)
    
    return processed_data

test_df = pd.read_csv('/kaggle/input/titanic/test.csv')  

test_data_processed = preprocess_data(test_df)


# In[ ]:



y_pred = xgb_clf.predict(test_data_processed)


# In[ ]:


# Prepare the submission DataFrame
submission_df = pd.DataFrame({
    'PassengerId': test_df['PassengerId'],
    'Survived': y_pred
})

submission_df = submission_df[['PassengerId', 'Survived']]

submission_df.to_csv('submission.csv', index=False)


# In[ ]:





# In[ ]:





# In[ ]:




