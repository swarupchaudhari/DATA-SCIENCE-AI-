import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

#Load dataset
dataset = pd.read_csv(r"C:\Users\Dell\Downloads\Salary_Data.csv")

#CHeck the shape of the dataset 
print("Dataset Shape:",dataset.shape) #(30,2)

x = dataset.iloc[:,:-1]
y = dataset.iloc[:,-1]

from sklearn.model_selection  import train_test_split 
x_train, x_test,y_train,y_test = train_test_split(x,y, test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)
print(y_pred)

comparison = pd.DataFrame({'Actual':y_test,'Predicted':y_pred})
print(comparison)

plt.scatter(x_test,y_test,color='pink')
plt.plot(x_train,regressor.predict(x_train),color='green')
plt.title("Salary VS Experience (Test Set)")
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Validation or future data 

c_inter = regressor.intercept_
print(f'intercept: {regressor.intercept_}')

m_coef = regressor.coef_
print(f'Coefficient : {regressor.coef_}')

model_coef= regressor.coef_
print(model_coef)

model_const = regressor.intercept_
print(model_const)

y_12 = model_coef * 12 + model_const
print(y_12)

y_20 = model_coef * 20 + model_const
print(y_20)

bias_training = regressor.score(x_train,y_train)
print(bias_training)

variance_testing = regressor.score(x_test, y_test)
print(variance_testing)


#Lets implement stats to this model 

dataset.mean() # this will give us mean of that particular 
dataset['Salary'].mean()
dataset['YearsExperience'].mean()

dataset.median() # this will give median of entire dataframe 
dataset['Salary'].median()
dataset['YearsExperience'].median()


dataset.var() # this will give variance of entire dataframe 
dataset['Salary'].var()
dataset['YearsExperience'].var()

dataset.std() # this will give standard deviation of enitre dataframe 
dataset['Salary'].std()
dataset['YearsExperience'].std()

# for calculating cv we have to import a library first
from scipy.stats import variation 
variation(dataset.values) #this will give cv of entire dataframe 
variation(dataset['Salary']) # this will give us cv of that particular column 
variation(dataset['YearsExperience'])

dataset.corr() #this will give correlation of enitre dataframe 

dataset['Salary'].corr(dataset['YearsExperience']) # this will give us correlation  between these salary and experience  (these two attributes)
dataset['Salary'].corr(dataset['Salary'])

# Skweness 
dataset.skew() # this will give skewness of entire dataframe 
dataset['Salary'].skew() #this will give us skewness of that particular column 

# Standard Error
dataset.sem() # this will give standard error of entire dataframe 
dataset['Salary'].sem() #this will give us standard error of that particular error 


# Z-SCORE 

# for calculating  Z-Score we have to import a library first 
import scipy.stats as stats
dataset.apply(stats.zscore) #this will give Z-score of entire dataframe

stats.zscore(dataset['Salary'])  # this wil give us Z-Score of that particular column 
stats.zscore(dataset['YearsExperience']) 



# ANOVA

# SSR 
y_mean=np.mean(y) 
SSR = np.sum((y_pred-y_mean)**2)
print(SSR)

#SSE
y=y[0:6]
SSE=np.sum((y-y_pred)**2)
print(SSE)

# SST 
mean_total = np.mean(dataset.values)

# here df.to_numpy()will convert pandas Dataframe to Nump
SST=np.sum((dataset.values-mean_total)**2)
print(SST)

# R2(Adjusted R2 )
r_square = 1 - (SSR / SST)
r_square
print(r_square)


bias = regressor.score(x_train, y_train)
print(bias)

variance = regressor.score(x_test, y_test)
print(variance)

# Degree of Freedom
a = dataset.shape[0]  #this will gives no.of rows 
b = dataset.shape[1]  #this will gives us no of columns 

degree_of_freedom = a - b 
print(degree_of_freedom) # this will give us degree of freedom of entire dataset


# SUM OF SQUARES REGRESSION (SSR)

# First we have to separate dependent and independent variables 
X = dataset.iloc[:,:-1].values # independent variable 
y = dataset.iloc[:,1].values # dependent variable 

y_mean = np.mean(y) # this will calculate mean of dependent variable 

from sklearn.model_selection import train_test_split 
X_train , X_test, y_train , y_test = train_test_split(X,y, test_size=0.20,random_state=0)

from sklearn.linear_model import LinearRegression 
reg = LinearRegression()
reg.fit(X_train, y_train)
y_predict = reg.predict(X_test) # before doing this we have to train , test and split 


SSR = np.sum((y_predict - y_mean) ** 2)
print(SSR)


# SUM OF SQUARES ERROR (SSE)


# First we have to separate dependent and independent variables 
X = dataset.iloc[:,:-1].values # independent variable 
y = dataset.iloc[:,1].values # dependent variable 

y_mean = np.mean(y) # this will calculate mean of dependent variable 

from sklearn.model_selection import train_test_split 
X_train , X_test, y_train , y_test = train_test_split(X,y, test_size=0.20,random_state=0)

from sklearn.linear_model import LinearRegression 
reg = LinearRegression()
reg.fit(X_train, y_train)
y_predict = reg.predict(X_test) # before doing this we have to train , test and split 
y = y[0:6]

SSE= np.sum((y-y_predict) ** 2)
print(SSE)


#ml developer 


import pickle 

#Save the trained model to disk 
filename ='linear_regression_model.pkl'

#Open a file in write-binary mode and dump the model 
with open (filename,'wb')   as file :
     pickle.dump(regressor, file )
     
     print("Model has been pickled and saved as linear_regression_model.pkl")


