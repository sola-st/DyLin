#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "/kaggle/input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # Problema da semana 1  

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.plots import plot_convergence


# ### Controle
# baseline 76.555%

# ## Carga de Dados de treino e teste

# In[ ]:


train = pd.read_csv('/kaggle/input/titanic/train.csv')
test = pd.read_csv('/kaggle/input/titanic/test.csv')


# In[ ]:


train.head()


# In[ ]:


test.head()


# ## Descrição estátistica das features núméricas

# In[ ]:


train.describe()


# In[ ]:


train.info()


# ## Verificar valores nulos ou NAN

# In[ ]:




# In[ ]:




# ## Mapear as colunas

# In[ ]:


train.columns


# In[ ]:


X_train = train.drop(['PassengerId', 'Survived'], axis = 1)

X_test = test.drop(['PassengerId'], axis = 1)


# ## Criar features

# In[ ]:


def criar_features(X):
    sexo = {'female':1, 'male':0}
    X['mulher'] = X['Sex'].map(sexo)

    X['Fare'] = X['Fare'].fillna(X['Fare'].mean())

    X['Age'] = X['Age'].fillna(X['Age'].mean())
#     # Calcular a mediana de 'Age' por grupo de 'Sex' e 'Pclass' e preencher valores ausentes
#     X['Age'] = X.groupby(['Sex', 'Pclass'])['Age'].transform(lambda x: x.fillna(x.median()))
# Criar feature 'FamilySize' (Tamanho da Família)
    X['FamilySize'] = X['SibSp'] + X['Parch'] + 1

    # Preencher valores ausentes de 'Embarked' com 'S' e mapear para numérico
    X['Embarked'] = X['Embarked'].fillna('S')
    portos = {'S': 1, 'C': 2, 'Q': 3}
    X['porto'] = X['Embarked'].map(portos)

    # Criar feature 'crianca' para indicar se a idade é menor que 12
    X['crianca'] = np.where(X['Age'] < 12, 1, 0)

    # Criar feature 'IsAlone' (Está Sozinho)
    X['IsAlone'] = np.where(X['FamilySize'] == 1, 1, 0)

    # Extrair títulos dos nomes e mapear para numérico
    X['Title'] = X['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    titulos = {
        'Mr': 1, 
        'Miss': 2, 'Ms': 2, 'Mlle': 2,
        'Mrs': 3, 'Mme': 3,
        'Master': 4,
        'Rare': 5      
    }
    X['Title'] = X['Title'].map(titulos)
    X['Title'] = X['Title'].fillna(0)
    
    # Criar feature 'cabinknow' (Conhece a Cabine)
    X['cabinknow'] = np.where(X['Cabin'].isnull(), 0, 1)

    # Criar feature 'FarePerPerson' (Tarifa por Pessoa)
    X['FarePerPerson'] = X['Fare'] / X['FamilySize']

    return X

# Aplicar a função nos conjuntos de treino e teste
X_train = criar_features(X_train)
X_test = criar_features(X_test)


# ## Selecionar as features

# In[ ]:


features = ['Pclass', 
            'Age', 
            'SibSp',
            'Parch', 
            'Fare', 
            'mulher', 
            'porto', 
            'crianca',
#             'IsAlone', 
#             'Title',
#             'cabinknow', 
#             'FarePerPerson'
           ]

X_train = X_train[features]
X_test = X_test[features]

y_train = train['Survived']


# ## Visualização

# In[ ]:


import matplotlib.pyplot as plt

for i in X_train.columns:
    plt.hist(X_train[i])
    plt.title(i)
    plt.show()


# ## Groupy

# In[ ]:


train.groupby(['Survived']).count()


# ## pivot_table

# In[ ]:


pd.pivot_table(train, index = ['Survived'], columns = ['Pclass'], values = 'PassengerId', aggfunc = 'count')


# ## Padronização das features

# In[ ]:


scaler = StandardScaler() #media 0 e desvio padrão 1

X_train_sc = scaler.fit_transform(X_train)

X_test_sc = scaler.transform(X_test)


# ## Modelo de validação cruzada

# ### Logistic Regression

# In[ ]:


model_lr = LogisticRegression(random_state=0, 
                              n_jobs = -1 )
score =  cross_val_score(model_lr,X_train_sc, y_train, cv = 10, n_jobs = -1)

np.mean(score)


# In[ ]:


def treinar_modelo_lr(parametros):
    model_lr = LogisticRegression(C=parametros[0], 
                                  solver=parametros[1], 
                                  max_iter=parametros[2], 
                                  random_state=0)
    score = cross_val_score(model_lr, X_train_sc, y_train, cv=10)
    mean_score = np.mean(score)
    return -mean_score

parametros_lr = [(1e-4, 10),                  # C
                 ('liblinear', 'lbfgs'),       # solver
                 (50, 300)]                    # max_iter

otimos_lr = gp_minimize(treinar_modelo_lr, parametros_lr, random_state=0, verbose=1, n_calls=50, n_random_starts=10)
plot_convergence(otimos_lr)
plt.show()


# In[ ]:


model_lr = LogisticRegression(C=otimos_lr.x[0], 
                              solver=otimos_lr.x[1], 
                              max_iter=otimos_lr.x[2], 
                              random_state=0)
score = cross_val_score(model_lr, X_train_sc, y_train, cv=10)
np.mean(score)


# ### Naive Bayes para Classificação

# In[ ]:


from sklearn.naive_bayes import GaussianNB

model_nb = GaussianNB()

score = cross_val_score(model_nb, X_train_sc, y_train, cv = 10)

np.mean(score)


# In[ ]:


def treinar_modelo_gnb(parametros):
    model_gnb = GaussianNB(var_smoothing=parametros[0])
    score = cross_val_score(model_gnb, X_train_sc, y_train, cv=10)
    mean_score = np.mean(score)
    return -mean_score

parametros_gnb = [(1e-9, 1e-1)]  # var_smoothing

otimos_gnb = gp_minimize(treinar_modelo_gnb, parametros_gnb, random_state=0, verbose=1, n_calls=30, n_random_starts=10)
plot_convergence(otimos_gnb)
plt.show()


# In[ ]:


model_gnb = GaussianNB(var_smoothing=otimos_gnb.x[0])
score = cross_val_score(model_gnb, X_train_sc, y_train, cv=10)
np.mean(score)


# ### KNN para classificação

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier

model_knn = KNeighborsClassifier(n_neighbors= 5, p = 2, n_jobs = -1 )

score = cross_val_score(model_knn, X_train_sc, y_train, cv = 10)

np.mean(score)


# In[ ]:


def treinar_modelo_knn(parametros):
    model_knn = KNeighborsClassifier(n_neighbors=parametros[0], 
                                     weights=parametros[1], 
                                     p=parametros[2], 
                                     n_jobs=-1)
    score = cross_val_score(model_knn, X_train_sc, y_train, cv=10)
    mean_score = np.mean(score)
    return -mean_score

parametros_knn = [(1, 30),               # n_neighbors
                  ('uniform', 'distance'), # weights
                  (1, 2)]                 # p (Minkowski metric, 1 for Manhattan, 2 for Euclidean)

otimos_knn = gp_minimize(treinar_modelo_knn, 
                         parametros_knn, 
                         random_state=0, 
                         verbose=1, 
                         n_calls=30, 
                         n_random_starts=10)
plot_convergence(otimos_knn)
plt.show()


# In[ ]:


model_knn = KNeighborsClassifier(n_neighbors=otimos_knn.x[0], 
                                 weights=otimos_knn.x[1], 
                                 p=otimos_knn.x[2], 
                                 n_jobs=-1)
score = cross_val_score(model_knn, X_train_sc, y_train, cv=10)
np.mean(score)


# ### SVM para classificação

# In[ ]:


from sklearn.svm import SVC

model_svc = SVC(C = 3, 
                kernel = 'rbf', 
                degree = 2, 
                gamma = 0.1)

score = cross_val_score(model_svc, X_train_sc, y_train, cv = 10)

np.mean(score)


# In[ ]:


# Função de treinamento e avaliação do modelo
def treinar_modelo_svc(parametros):
    model_svc = SVC(
        C=parametros[0], 
        kernel=parametros[1], 
        gamma=parametros[2], 
        probability=True,
        random_state=0
    )
    score = cross_val_score(model_svc, X_train_sc, y_train, cv=10, n_jobs=-1)
    mean_score = np.mean(score)
    return -mean_score

# Espaço de busca para hiperparâmetros
parametros_svc = [
    Real(1e-3, 10, "log-uniform"),            # C em escala log-uniforme
    Categorical(['linear', 'rbf', 'poly']),   # kernel
    Real(1e-6, 1e+1, "log-uniform")           # gamma em escala log-uniforme
]

# Otimização dos hiperparâmetros com gp_minimize
otimos_svc = gp_minimize(
    treinar_modelo_svc, 
    parametros_svc, 
    random_state=0, 
    verbose=1, 
    n_calls=10, 
    n_random_starts=10
)

# Plot da convergência
plot_convergence(otimos_svc)
plt.show()

# Resultados da otimização



# In[ ]:


model_svc = SVC(C=otimos_svc.x[0], 
                kernel=otimos_svc.x[1], 
                gamma=otimos_svc.x[2], 
                probability=True,
                random_state=0)
score = cross_val_score(model_svc, X_train_sc, y_train, cv=10)
np.mean(score)


# ### Decision Tree

# In[ ]:


from sklearn.tree import DecisionTreeClassifier

model_dt = DecisionTreeClassifier(criterion = 'entropy', 
                                  max_depth = 3, 
                                  min_samples_split = 2, 
                                  min_samples_leaf = 1, 
                                  random_state = 0 )

score = cross_val_score(model_dt, X_train_sc, y_train, cv = 10)

np.mean(score)


# In[ ]:


def treinar_modelo_dt(parametros):
    model_dt = DecisionTreeClassifier(criterion=parametros[0], 
                                      max_depth=parametros[1], 
                                      min_samples_split=parametros[2], 
                                      min_samples_leaf=parametros[3], 
                                      random_state=0)
    score = cross_val_score(model_dt, X_train_sc, y_train, cv=10)
    mean_score = np.mean(score)
    return -mean_score

parametros_dt = [('entropy', 'gini'),  # criterion
                 (3, 20),              # max_depth
                 (5, 15),              # min_samples_split
                 (1, 10)]              # min_samples_leaf

otimos_dt = gp_minimize(treinar_modelo_dt, parametros_dt, random_state=0, verbose=1, n_calls=50, n_random_starts=10)
plot_convergence(otimos_dt)
plt.show()


# In[ ]:


model_dt = DecisionTreeClassifier(criterion=otimos_dt.x[0], 
                                  max_depth=otimos_dt.x[1], 
                                  min_samples_split=otimos_dt.x[2], 
                                  min_samples_leaf=otimos_dt.x[3], 
                                  random_state=0)
score = cross_val_score(model_dt, X_train_sc, y_train, cv=10)
np.mean(score)


# ### Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

model_rf = RandomForestClassifier(criterion = 'entropy', 
                                  n_estimators = 100, 
                                  max_depth = 3, 
                                  min_samples_split = 2, 
                                  min_samples_leaf = 1, 
                                  random_state = 0, 
                                  n_jobs = -1 )

score = cross_val_score(model_rf, X_train_sc, y_train, cv = 10)

np.mean(score)


# #### Otimização de hiperparametros

# In[ ]:


def treinar_modelo(parametros):
    model_rf = RandomForestClassifier(criterion = parametros[0], 
                                      n_estimators = parametros[1], 
                                      max_depth = parametros[2], 
                                      min_samples_split = parametros[3], 
                                      min_samples_leaf = parametros[4], 
                                      random_state = 0, 
                                      n_jobs = -1 )
    score = cross_val_score(model_rf, X_train_sc, y_train, cv = 10)
    mean_score = np.mean(score)
    return - mean_score

parametros = [('entropy', 'gini'), 
              (500, 1200), 
              (3, 20),
              (5, 15),
              (1, 10)]

otimos = gp_minimize(treinar_modelo, 
                     parametros, 
                     random_state = 0, 
                     verbose = 1, 
                     n_calls = 30, 
                     n_random_starts = 10 , 
                     n_jobs = -1 )

plot_convergence(otimos)
plt.show()


# In[ ]:


model_rf = RandomForestClassifier(criterion = otimos.x[0], 
                                  n_estimators = otimos.x[1], 
                                  max_depth = otimos.x[2], 
                                  min_samples_split = otimos.x[3], 
                                  min_samples_leaf = otimos.x[4], 
                                  random_state = 0, 
                                  n_jobs = -1 )
score = cross_val_score(model_rf, X_train_sc, y_train, cv = 10)

np.mean(score)


# ## Ensanble model (Voting)

# In[ ]:


from sklearn.ensemble import VotingClassifier

# Configurando o ensemble VotingClassifier
model_voting = VotingClassifier(
    estimators=[
        ('LR', model_lr), 
#         ('NB', model_nb), 
        ('KNN', model_knn), 
        ('SVC', model_svc), 
        ('RF', model_rf)
    ],
    voting='hard'  # Use 'soft' para ponderação de probabilidades
)

# Treinando o ensemble VotingClassifier
model_voting.fit(X_train_sc, y_train)

# Avaliação do ensemble com validação cruzada
score = cross_val_score(model_voting, X_train_sc, y_train, cv=10, n_jobs=-1)  # Aqui n_jobs está correto


# ## Modelo Final

# In[ ]:


model_voting.fit(X_train_sc, y_train)

y_pred = model_voting.predict(X_train_sc)

confusion_matrix(y_train, y_pred)


# In[ ]:


score = model_voting.score(X_train_sc, y_train)
score


# ## Predição dos dados de teste

# In[ ]:


y_pred_test = model_voting.predict(X_test_sc)

submission = pd.DataFrame(test['PassengerId'])

submission['Survived'] = y_pred_test

submission.to_csv('submission.csv', index= False)


# In[ ]:


submission


# In[ ]:




