# (05)***************Multiple Linear Regression***************

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = r'datasets/M_Regression.csv'
dataset = pd.read_csv(path)
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,3].values

# ---for spliting the data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)

# ----multiple Linear Regression

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,y_train)
y_predict = model.predict(X_test)
# print(y_predict)
# print(y_test)
