# (02)****************Data preprocessing in ML ALGO*****************]

# ---pre_processing refers to the transformation applied to our data before feeding it to the algorithm
# ---Data preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is nor feasible for the analysis.


# ----for filling the values of dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('datasets/dataset.csv')
# print(dataset.head())
X=dataset.iloc[:,:-1].values
# X=dataset.drop(columns='Married',axis=1)
y=dataset.iloc[:,3]
# y=dataset['Married']

# ----Now filling the data using sklearn.preprocessing
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values =np.nan,strategy = 'mean')
imputer=imputer.fit(X[:,1:3])
X=imputer.transform(X[:,1:3])

# ----Now converting categorical data to numarical data
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
X[:,0] = labelencoder.fit_transform(X[:,0])

# ----for seperating the column names and also converting the categorical data into numarical data we use this library
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False)
X=ohe.fit_transform(X)


# ----Now spliting the data into train and test data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=12)


# ----Now we apply the feature scalling in the dataset
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_test=sc.fit_transform(X_test)
X_train=sc.fit_transform(X_train)




