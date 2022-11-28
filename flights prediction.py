#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 26 14:35:15 2022

@author: yihenglin
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns

flights = pd.read_csv('Clean_Dataset.csv').drop(columns='Unnamed: 0')


flights.hist(bins=50, figsize=(20,15))
plt.show()

corr = flights.corr()
mask = np.triu(np.ones_like(corr,dtype = bool))

plt.figure(dpi=100)
plt.title('Correlation Analysis')
sns.heatmap(corr,mask=mask,annot=False,lw=0,linecolor='white',cmap='magma',fmt = "0.2f")
plt.xticks(rotation=90)
plt.yticks(rotation = 0)
plt.show()


"""
flights_num = flights.select_dtypes(include=[np.number]).drop("price", axis=1)
flights_cat = flights.select_dtypes(include=["object"])

from sklearn.preprocessing import OneHotEncoder

cat_encoder = OneHotEncoder()
flights_cat_1hot = cat_encoder.fit_transform(flights_cat)
#print(flights_cat_1hot)

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")
imputer.fit(flights_num)
#print(imputer.statistics_)
#X = imputer.transform(flights_num)
#print(X)


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('std_scaler', StandardScaler()),
    ])

from sklearn.compose import ColumnTransformer

num_attribs = list(flights_num)
cat_attribs = list(flights_cat)
print(num_attribs)
print(cat_attribs)

full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])

flights_prepared = full_pipeline.fit_transform(flights)
X_test = full_pipeline.fit_transform(X_test)
X_train = full_pipeline.fit_transform(X_train)
print(flights.shape)
print(flights_prepared.shape)
print(X_test.shape)
print(X_train.shape)
#print(flights_prepared)

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

from sklearn.metrics import mean_squared_error
from sklearn.linear_model import SGDRegressor
mySGDModel = SGDRegressor()
mySGDModel.fit(X_train,y_train)
y_predict = mySGDModel.predict(X_test)
mse = mean_squared_error(y_test, y_predict)
mySGDModel_rmse = np.sqrt(mse)
print(mySGDModel_rmse)
"""

flights = pd.read_csv("Clean_Dataset.csv").drop(columns='Unnamed: 0')

flights_copy = flights.copy()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for col in flights_copy.columns:
    if flights_copy[col].dtype == 'object':
        flights_copy[col] = le.fit_transform(flights_copy[col])
print(flights_copy.head(20))

flights_copy.hist(bins=50, figsize=(20,15))
plt.show()

corr = flights_copy.corr()
mask = np.triu(np.ones_like(corr,dtype = bool))

plt.figure(dpi=100)
plt.title('Correlation Analysis')
sns.heatmap(corr,mask=mask,annot=False,lw=0,linecolor='white',cmap='magma',fmt = "0.2f")
plt.xticks(rotation=90)
plt.yticks(rotation = 0)
plt.show()


from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(flights_copy, test_size=0.2, random_state=42)
X_train = train_set.drop("price", axis=1);
y_train = train_set["price"].copy();
X_test = test_set.drop("price", axis=1);
y_test = test_set["price"].copy();

from sklearn.preprocessing import MaxAbsScaler
mascaler = MaxAbsScaler()
X_train = mascaler.fit_transform(X_train)
X_test = mascaler.fit_transform(X_test)
print(X_test)
X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)
print(X_train)

print(y_test)
print(X_test)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred_lr = lin_reg.predict(X_test)
print(y_pred_lr)
lin_rmse = np.sqrt(mean_squared_error(y_test, y_pred_lr))
print(lin_rmse)
score = lin_reg.score(X_test, y_test)  
print(score)  
print(cross_val_score(lin_reg, X_train, y_train))


tree_reg = DecisionTreeRegressor(random_state = 42)
tree_reg.fit(X_train, y_train)
y_pred_dt = tree_reg.predict(X_test)
print(y_pred_dt)
tree_rmse = np.sqrt(mean_squared_error(y_test, y_pred_dt))
print(tree_rmse)
score = tree_reg.score(X_test, y_test)  
print(score) 
print(cross_val_score(tree_reg, X_train, y_train))


forest_reg = RandomForestRegressor(n_estimators=100, random_state=42)
forest_reg.fit(X_train, y_train)
y_pred_rf = forest_reg.predict(X_test)
print(y_pred_rf)
forest_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))
print(forest_rmse)
score = forest_reg.score(X_test, y_test)  
print(score)  
print(cross_val_score(forest_reg, X_train, y_train))


mySGDModel = SGDRegressor()
mySGDModel.fit(X_train,y_train)
y_predict = mySGDModel.predict(X_test)
print(y_predict)
mse = mean_squared_error(y_test, y_predict)
mySGDModel_rmse = np.sqrt(mse)
print(mySGDModel_rmse)
score = mySGDModel.score(X_test, y_test)  
print(score)  
print(cross_val_score(mySGDModel, X_train, y_train))
