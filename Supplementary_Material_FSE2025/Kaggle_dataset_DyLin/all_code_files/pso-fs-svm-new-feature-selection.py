#!/usr/bin/env python
# coding: utf-8

# # Model Explanation with Emphasis on the Power of PSO (Particle Swarm Optimization)
# 
# ## ① Data Preprocessing
# 
# The `preprocess_data` function prepares the foundational data for the model. At this stage, PSO has not yet been applied. This step handles missing values, encodes categorical variables, and standardizes the data, setting a solid groundwork for the SVM model.
# 
# ## ② Feature Selection and Hyperparameter Optimization Using PSO
# 
# This is where PSO truly excels. Traditionally, feature selection and hyperparameter tuning are performed separately, but in this model, PSO is utilized to optimize both aspects simultaneously.
# 
# ### Key Advantages of PSO
# 
# - **Global Search Capability**: PSO is highly effective at exploring all potential combinations of features and hyperparameters globally. This reduces the risk of falling into local minima and ensures the SVM model is more accurate and reliable.
# 
# - **Efficient Computation**: PSO systematically searches a broad space of features and parameters, finding near-optimal solutions efficiently. This is particularly advantageous for tuning the SVM's hyperparameters, which can be challenging to optimize manually.
# 
# - **Dynamic Adaptation**: As particles in PSO converge towards the optimal solution, the search range narrows, enhancing search precision and accelerating convergence. This results in high performance achieved in a shorter time frame.
# 
# ## ③ The PSO Optimization Process
# 
# The `pso_optimization` function embodies the core of PSO's optimization capabilities, simultaneously refining feature selection and hyperparameter tuning.
# 
# - **Parallel Exploration**: PSO assesses multiple candidate solutions (particles) in parallel, evaluating various combinations of features and hyperparameters simultaneously. This parallel exploration contributes to finding the best-performing model efficiently.
# 
# - **Leveraging Collective Knowledge**: Each particle shares its search experience with the swarm, creating a collective intelligence that guides the entire swarm towards the optimal solution. This shared knowledge increases the likelihood of converging on a superior model.
# 
# ## ④ Re-training the Model with Optimized Parameters
# 
# Once PSO identifies the best features and hyperparameters, the SVM model is re-trained using these optimized parameters. At this stage, the model is expected to deliver superior performance, having been finely tuned by PSO.
# 
# ## ⑤ Saving Prediction Results
# 
# The re-trained model is then used to generate predictions on the test data, and these predictions are saved. Thanks to PSO's optimization, the predictions from the SVM model are anticipated to be more accurate than those from models optimized using traditional techniques.
# 
# PSO is a powerful tool for optimizing machine learning models. In this scenario, it effectively enhances both feature selection and hyperparameter tuning for the SVM model, resulting in more precise and reliable predictions. By leveraging PSO’s global search capabilities and adaptive convergence, the model achieves superior accuracy and efficiency.
# 

# In[ ]:


#get_ipython().system('pip install pyswarm')


# In[ ]:


import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.impute import SimpleImputer
from pyswarm import pso

# Data preprocessing
def preprocess_data(train_path, test_path):
    # Load data from specified paths
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    
    # Handle missing values
    imputer = SimpleImputer(strategy='median')
    train_data['Age'] = imputer.fit_transform(train_data[['Age']])
    test_data['Age'] = imputer.transform(test_data[['Age']])

    train_data['Fare'] = imputer.fit_transform(train_data[['Fare']])
    test_data['Fare'] = imputer.transform(test_data[['Fare']])

    train_data['Embarked'] = train_data['Embarked'].fillna(train_data['Embarked'].mode()[0])
    test_data['Embarked'] = test_data['Embarked'].fillna(test_data['Embarked'].mode()[0])

    # Convert categorical variables to numerical
    label_encoder = LabelEncoder()
    train_data['Sex'] = label_encoder.fit_transform(train_data['Sex'])
    test_data['Sex'] = label_encoder.transform(test_data['Sex'])
    train_data['Embarked'] = label_encoder.fit_transform(train_data['Embarked'])
    test_data['Embarked'] = label_encoder.transform(test_data['Embarked'])

    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    X_train = train_data[features]
    y_train = train_data['Survived']
    X_test = test_data[features]

    # Standardize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, y_train, X_test

# Define the objective function
def objective_function(params, X, y):
    C = params[0]
    gamma = params[1]
    feature_mask = params[2:]

    # Apply mask to select features
    selected_features = X[:, feature_mask.astype(bool)]

    # Initialize SVC model and perform cross-validation
    svc = SVC(C=C, gamma=gamma, kernel='rbf', probability=True)
    scores = cross_val_score(svc, selected_features, y, cv=5, scoring='accuracy')
    
    # Return the negative accuracy (for PSO minimization)
    return -np.mean(scores)

# PSO optimization
def pso_optimization(X, y, num_features, random_seed):
    np.random.seed(random_seed)
    # Set PSO parameters
    lb = [0.1, 0.001] + [0] * num_features  # C, gamma, feature_mask
    ub = [10, 1] + [1] * num_features
    
    # Run PSO
    optimal_params, fopt = pso(objective_function, lb, ub, args=(X, y), swarmsize=20, maxiter=50, debug=True)
    
    return optimal_params, fopt

# Main process
if __name__ == "__main__":
    # Load and preprocess data
    train_path = '/kaggle/input/titanic/train.csv'
    test_path = '/kaggle/input/titanic/test.csv'
    X_train, y_train, X_test = preprocess_data(train_path, test_path)

    # Perform PSO x times with different initial conditions
    best_params = None
    best_score = float('inf')
    x=1

    for i in range(x):
        random_seed = np.random.randint(0, 10000)
        optimal_params, fopt = pso_optimization(X_train, y_train, X_train.shape[1], random_seed)
        print(f"Run {i+1}: Optimal Parameters = {optimal_params}, Score = {-fopt}")
        
        if -fopt < best_score:
            best_score = -fopt
            best_params = optimal_params

    print(f"Best Optimal Parameters: {best_params}, Best Score = {best_score}")

    # Re-train SVC model with the best optimal parameters and feature set
    C_opt = best_params[0]
    gamma_opt = best_params[1]
    feature_mask_opt = best_params[2:].astype(bool)
    selected_features = X_train[:, feature_mask_opt]

    svc_opt = SVC(C=C_opt, gamma=gamma_opt, kernel='rbf', probability=True)
    svc_opt.fit(selected_features, y_train)

    # Make predictions on the test data
    X_test_selected = X_test[:, feature_mask_opt]
    predictions = svc_opt.predict(X_test_selected)

    # Output predictions
    output = pd.DataFrame({'PassengerId': pd.read_csv(test_path)['PassengerId'], 'Survived': predictions})
    output.to_csv('/kaggle/working/titanic_svc_pso_predictions.csv', index=False)

