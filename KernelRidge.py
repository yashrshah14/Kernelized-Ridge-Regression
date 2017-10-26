import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from math import sqrt
dataset=pd.read_csv("C:\Users\Yash\Downloads")
dataset["BIAS"]=1
X=dataset.values
#X=sklearn.preprocessing.normalize(dataset)
#print(X)
response=pd.read_csv("C:\Users\Yash\Downloads")
#Y=sklearn.preprocessing.normalize(response)
Y=response.values
X=X[0:800,:]
Y=Y[0:800,:]
#Y=np.ravel(Y)
X=np.asmatrix(X)
Y=np.asmatrix(Y)
print(X[0])
print(X[1])
print(X[0]*np.transpose(X[1]))

def linear_kernel(x,x_prime):
    return (np.transpose(x).dot(x_prime))

def poly_kernel(x,x_prime,gamma,r,m):
    return (float((gamma*(np.transpose(x)*(x_prime))+r)**m))

def gaussian_kernel(x,x_prime,sigma):
    return (np.exp(-sum((x-x_prime)**2)/2*sigma**2))

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1)
from sklearn.kernel_ridge import KernelRidge
grid_linear=GridSearchCV(KernelRidge(kernel='linear'), param_grid={'alpha':[1e0,0.1,1e-2,1e-3,0.5,2,3,5]}, cv=5)
grid_linear.fit(X_train,Y_train)
grid_linear.best_score_
grid_linear.best_params_
y_pred_lin=grid_linear.predict(X_test)
score=sklearn.metrics.r2_score(Y_test, y_pred_lin)
print(score)
grid_polynomial= GridSearchCV(KernelRidge(kernel="polynomial", degree=3), cv=5, param_grid={"alpha":[1e0,0.1,1e-2,1e-3,0.5,2,3,5], "gamma":[0.001,0.0001,0.01,0.1,1,0.5,2,3]})
grid_polynomial.fit(X_train,Y_train)
grid_polynomial.best_params_
y_pred_poly=grid_polynomial.predict(X_test)
score_poly=sklearn.metrics.r2_score(Y_test, y_pred_poly)
print(score_poly)
grid_gaussian= GridSearchCV(KernelRidge(kernel="rbf"), cv=5, param_grid={"alpha":[1e0,0.1,1e-2,1e-3,0.5,2,3,5], "gamma":[0.001,0.0001,0.01,0.1,1,0.5,2,3,10]})
grid_gaussian.fit(X_train,Y_train)
grid_gaussian.best_params_
y_pred_gauss=grid_gaussian.predict(X_test)
score_gauss=sklearn.metrics.r2_score(Y_test, y_pred_gauss)
print(score_gauss)
