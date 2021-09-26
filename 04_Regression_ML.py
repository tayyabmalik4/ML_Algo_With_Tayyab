# (04)*****************Regression in Machine learning*****************
# ----importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('datasets/Salary_DataSet.csv')
X = dataset.iloc[:,:-1].values
y= dataset.iloc[:,1].values
# print(dataset.head())

# ----spliting the data into 4 parts
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)

# ---Now Appling the Regression line
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,y_train)
y_predict = model.predict(X_test)
print(y_predict)
print(y_test)


# ----Now Visualizing the values through graph
# plt.scatter(X_train,y_train,color='red')
# plt.plot(X_train,model.predict(X_train),color='b')
# plt.title("Linear Regression Salary Vs Experience")
# plt.xlabel("Years of Employee")
# plt.ylabel("Salaries of employees")

plt.scatter(X_test,y_test,color='red')
plt.plot(X_test,model.predict(X_test),color='b')
plt.title("Linear Regression Salary Vs Experience")
plt.xlabel("Years of Employee")
plt.ylabel("Salaries of employees")
plt.show()