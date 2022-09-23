import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from grad_des import gd
from grad_des import cost
df=pd.read_csv('Salary_Data.csv')
X=np.array(df['YearsExperience']).reshape(-1,1)
y=np.array(df['Salary']).reshape(-1,1)
theta=np.zeros((2,1))#initializing parameters to zeros
alpha=0.02#learning rate
mod_theta=gd(X,y,theta,alpha,6000)#function call of grad_des
print("By Gradient Descent :")
print("Intercept: {}".format(mod_theta[0,0]))
print("slope: {}".format(mod_theta[1,0]))#slope and intercept
plt.scatter(df[['YearsExperience']],df[['Salary']],label="Scatter",color='red')
A=np.c_[np.ones((30,1)),X]
plt.plot(X,np.dot(A,mod_theta),label="Best Fit")
plt.legend()
plt.xlabel("Years of Experience")
plt.ylabel("Salary in Dollar")
plt.show()
##using Sklearn
from sklearn import linear_model
reg=linear_model.LinearRegression()
reg.fit(X,y)
a=reg.intercept_
b=reg.coef_
print("By sklearn package :")
print("Intercept : {}".format(a[0]))
print("slope: {}".format(b[0,0]))
#######
## Normal equation ##
B=np.transpose(A)
C=(np.dot(B,A))
invC=np.linalg.inv(C)
E=np.dot(B,y)
print("By normal equation :")
print("intercept: {}".format(np.dot(invC,E)[0,0]))
print("slope: {}".format(np.dot(invC,E)[1,0]))
