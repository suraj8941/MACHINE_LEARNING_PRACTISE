import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

diabetes = datasets.load_diabetes()

#diabetes.keys() = ['data', 'target', 'frame', 'DESCR', 'feature_names', 'data_filename', 'target_filename'])
#print(diabetes.DESCR)

diabetes_X = diabetes.data                   ########## ONLY CHANGE IS HERE
diabetes_X_train = diabetes_X[:-30]
diabetes_X_test = diabetes_X[-30:]

diabetes_Y_train = diabetes.target[:-30]
diabetes_Y_test = diabetes.target[-30:]

model = linear_model.LinearRegression()
model.fit(diabetes_X_train, diabetes_Y_train)

diabetes_Y_predict = model.predict(diabetes_X_test)
print("Mean square error : ", mean_squared_error(diabetes_Y_test,diabetes_Y_predict))
print("Weight : ", model.coef_)
print("Intercept: ", model.intercept_)