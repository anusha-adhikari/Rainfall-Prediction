import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

data=pd.read_csv('/Users/macuser/Desktop/python + ML/archive (2)/rainfall in india 1901-2015 2.csv')

X=data.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13]].values
y=data.iloc[:,14].values

y=y.reshape(-1,1)

#cleansing the data
imp=SimpleImputer(missing_values=np.nan,strategy='most_frequent')
X=imp.fit_transform(X)
y=imp.fit_transform(y)

#encoding the strings
label=LabelEncoder()
X[:,0]=label.fit_transform(X[:,0])

#splitting of dataset
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

#training the model
reg=LinearRegression()
reg.fit(X_train,y_train)
y_pred=reg.predict(X_test)
figure,axis=plt.subplots(2)
axis[0].plot(y_test,color='green')
axis[0].set_title('Real values of annual rainfall')
axis[1].plot(y_pred,color='red')
axis[1].set_title('Predicted values of annual rainfall')
plt.show()

#obtaining the dataframe
df=np.concatenate((y_test,y_pred),axis=1)
dataframe=pd.DataFrame(df,columns=['Real annual rainfall','  predicted annual rainfall'])
print(dataframe)

#accuracy score
ac=reg.score(X_test, y_test)*100
print('Accuracy = ', ac)
