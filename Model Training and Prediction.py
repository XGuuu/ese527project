'''
This file contains the model training and testing part, including the cross-validation.
Since we have difficulty to predict the original test data set, we decide to split the 
original training data set into 
    (i) a new training data set, which is used for modeling, and 
    (ii) a validation data set, which is used for testing purpose.

In this file, we
1.  splited data set into training data set and test data set.
2.  performed the 5-fold validation in training data set.
3.  trained the following models: 
    LGBM Regressor, Random Forest regression, 
    Ridge regression, Lasso regression, and KNN regressor.
4.  performed the 5-fold validation and computed the mean of RMLSE.
5.  predict on the test data set and computed the mean of RMLSE.
6.  computed the time cost for each model.
'''
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
import preProcessing2 # use methods implemented in preProcessing2.py
import time

time_start = time.time() # start time

# read in pre-processed data
# one dataset for one type of meter reading
Train0, Train1, Train2, Train3 = preProcessing.getTrainData()

# split data into target column and feature columns
target0, features0 = preProcessing.splitData(Train0)
target1, features1 = preProcessing.splitData(Train1)
target2, features2 = preProcessing.splitData(Train2)
target3, features3 = preProcessing.splitData(Train3)

# generate polynomial and interaction features 
# generate a new feature matrix consisting of all polynomial combinations of the features with degree <= 2
ploy = PolynomialFeatures(2)
train_features0 = ploy.fit_transform(features0)
train_features1 = ploy.fit_transform(features1)
train_features2 = ploy.fit_transform(features2)
train_features3 = ploy.fit_transform(features3)

# split each data set into a training data set and a validation (test) data set
# each training data set or validation data set includes feature data and target value
train_X0, val_features0, train_target0, val_target0 = train_test_split(Train0, target0, random_state(0))
train_X1, val_features1, train_target1, val_target1 = train_test_split(Train1, target1, random_state(0))
train_X2, val_features2, train_target2, val_target2 = train_test_split(Train2, target2, random_state(0))
train_X3, val_features3, train_target3, val_target3 = train_test_split(Train3, target3, random_state(0))

# in training process, we try several regression models below
# LGBM Regressor, a GBDT (Gradient Boosting Decision Tree) model
lightgbm = LGBMRegressor(objective='regression', learning_rate=0.1, num_leaves=1024,
    feature_fraction=0.8,  bagging_fraction=0.8, bagging_freq=5)

# Random Forest regression
RandomRegression = RandomForestRegressor()

# Ridge regression
ridge = Ridge(alpha=0.5)

# Lasso regression
lasso = Lasso(alpha=0.5)

# Linear regression
lr = LinearRegression()

# KNN regressor
knn = KNeighborsRegressor()

# perform 5-fold validation, choose one model each time
kf = KFold(n_splits=5, random_state=10, shuffle=True)

featuress = [train_X0, train_X1, train_X2, train_X3]
targets = [train_target0, train_target1, train_target2, train_target3]
testss = [val_features0, val_features1, val_features2, val_features3]
test_targets = [val_target0, val_target1, val_target2, val_target3]

evaluate_target = []
evaluate_target.extend(val_target0.tolist())
evaluate_target.extend(val_target1.tolist())
evaluate_target.extend(val_target2.tolist())
evaluate_target.extend(val_target3.tolist())

prediction_target = []
for i in range(4):
    n = len(test_targets[i])
    regressor = lightgbm
    RMLSE_mean = []
    test_features = testss[i]
    test_target = test_targets[i]
    features = featuress[i]
    target = targets[i]
    for train_idx, val_idx in kf.split(features):

        train_features = features[train_idx]
        train_target = target[train_idx]
        val_features = features[val_idx]
        val_target = target[val_idx]

        length = len(val_target)

        regressor.fit(train_features, train_target)
        prediction = regressor.predict(val_features)
        prediction1 = (prediction + 1).tolist()
        val_target1 = (val_target+1).tolist()
        validation_RMSE = np.sqrt(mean_squared_error(prediction, val_target))
        for j in range(length):
            if prediction1[j] < 1:
                prediction1[j] = 1
        validation_RMLSE = np.sqrt((1/length)*np.sum(np.square(np.log(prediction1) - np.log(val_target1))))
        RMLSE_mean.append(validation_RMLSE)

    # compute and output the mean of validation RMLSE
    RMLSE_mean = np.mean(RMLSE_mean)
    print('The RMLSE during k ford validation is')
    print(RMLSE_mean)

    # predit test data
    prediction_test = regressor.predict(test_features)
    prediction_target.extend(prediction_test.tolist())
    prediction_test = (prediction_test + 1).tolist()
    target_test = (test_target + 1).tolist()
    for j in range(n):
        if prediction_test[j] < 1:
            prediction_test[j] = 1

    # compute and output the mean of test RMLSE
    test_RMLSE = np.sqrt((1 / n) * np.sum(np.square(np.log(prediction_test) - np.log(target_test))))
    print('the RMLSE in test dataset is')
    print(test_RMLSE)

# compute and output the mean of test RMLSE for all test data points
m = len(evaluate_target) 
prediction_target = [i+1 for i in prediction_target] # predicted test target value
evaluate_target = [i+1 for i in evaluate_target] # true test target value
for j in range(m):
    if prediction_target[j] < 1:
        prediction_target[j] = 1
RMLSE = np.sqrt((1 / m) * np.sum(np.square(np.log(prediction_target) - np.log(evaluate_target))))
print('over all RMLSE is')
print(RMLSE)

time_end = time.time() # end time

# compute and output time cost for the chosen model
print('time cost', time_end-time_start, 's')
