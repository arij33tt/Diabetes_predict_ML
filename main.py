# importing the library 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# load the data set 

data_set= pd.read_csv('ds.csv')
# print(data_set.shape)


#now seprating the data set into dependent and independent variables

X= data_set.iloc[:, :-1]
y=data_set.iloc[:,-1]

# scaler = StandardScaler()

# scaler.fit(X)

# X=scaler.transform(X)


 
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0,test_size=0.25,shuffle=True)


# this is for Support Vector Machine 

svm_class = SVC(kernel='linear')

svm_class.fit(X_train,y_train)

y_pred_svm =svm_class.predict(X_test)

accuracy_svm= accuracy_score(y_test,y_pred_svm)

print(accuracy_svm)

# now we will plot the graph

 

