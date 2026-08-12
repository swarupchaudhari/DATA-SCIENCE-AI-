import numpy as np 
import pandas as pd
import matplotlib.pyplot  as plt 

%matplotlib inline

dataset = pd.read_csv(r"C:\Users\Dell\Downloads\emp_sal.csv")

dataset.info()
dataset.head

X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# linear model  -- linear algor ( degree - 1)
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# polynomial model  ( bydefeaut degree - 2)

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=5)
X_poly = poly_reg.fit_transform(X)

poly_reg.fit(X_poly, y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)


# linear regression visualizaton 
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Linear Regression graph')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


# poly nomial visualization 

plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# predicton 

lin_model_pred = lin_reg.predict([[6.5]])
lin_model_pred

poly_model_pred = lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
poly_model_pred



# Decision Tree 

from sklearn.tree import DecisionTreeRegressor 
dt_reg = DecisionTreeRegressor(criterion = 'squared_error',splitter='best',max_depth=5)
dt_reg.fit(X,y)

dt_reg_pred = dt_reg.predict([[6.5]])
print(dt_reg_pred)


# KNN MODEL 

# KNN MODEL

from sklearn.neighbors import KNeighborsRegressor

knn_reg = KNeighborsRegressor(
    n_neighbors=3,
    p=1,
    weights='distance'
)

knn_reg.fit(X, y)

knn_reg_pred = knn_reg.predict([[6.5]])

print(knn_reg_pred)


# Random Forest
from sklearn.ensemble import RandomForestRegressor 
rf_reg = RandomForestRegressor(n_estimators=30,random_state=0)
rf_reg.fit(X,y)

rf_reg_pred =  rf_reg.predict([[6.5]])
print(rf_reg_pred)

from xgboost import XGBRFRegressor

import xgboost
print(xgboost.__version__)

# Create model
xgb_reg = XGBRFRegressor()

# Train model
xgb_reg.fit(X, y)

# Prediction
xgb_reg_pred = xgb_reg.predict([[6.5]])

print(xgb_reg_pred)











