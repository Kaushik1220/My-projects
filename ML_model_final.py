import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from pykalman import KalmanFilter
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

data=pd.read_csv(r'C:\Users\DELL\Documents\Academics\Iot project\Dataset\FINAL WEATHERDATA AMD.csv')
#Data analysing and dealing with null values
data.head()
data=data.dropna()
data
data=data.drop(columns='_key')
data=data.drop(columns='timestamp')
data['Dates'] = pd.to_datetime(data['time formatted']).dt.date
data['Time'] = pd.to_datetime(data['time formatted']).dt.time
#data['Dates']
#data['Time']
data['Dates']=data['Dates'].astype(str)
data['Time']=data['Time'].astype(str)
data[['Year','Month','Date']]=data['Dates'].str.split('-',expand=True)
data[['Hours','Min','Sec']]=data['Time'].str.split(':',expand=True)
#data.head()
data=data.drop(columns=['time formatted','Dates'])
#data.head()
data=data.drop(columns=['humidity','pressure'])
#data.head()
data=data.drop(columns='temperature')
#data.head()
#data
data=data.drop(columns='Time')
C:\Users\DELL\Documents\Academics\Iot project\Dataset\FINAL WEATHERDATA AMD.csvEDA
fig, ax = plt.subplots(figsize=(15,25))
sns.lineplot(data[1:8331],x='Date',y='temprature formatted')
fig.suptitle('Variation of temperature with days', fontsize=20)
plt.xlabel('Days',fontsize=18)
plt.ylabel('Temperature',fontsize=18)
plt.show()

fig, ax = plt.subplots(figsize=(15,25))

sns.lineplot(data=data[1:200],x='humidity formatted',y='temprature formatted')
plt.xlabel('Humidity',fontsize=20)
plt.ylabel('Temperature',fontsize=20)
plt.show()


plt.show()

fig, ax = plt.subplots(figsize=(15,25))
sns.lineplot(data=data[1:500],x='temprature formatted',y='Pressure Formatted')
plt.xlabel('Temperature')
plt.ylabel('Pressure')
plt.show()

fig, ax = plt.subplots(figsize=(15,25))
fig.suptitle('Variation of temperature with days', fontsize=20)

sns.boxplot(y=data['temprature formatted'])
plt.ylabel('Temperature',fontsize=18)
plt.show()
sns.boxplot(y=data['humidity formatted'])
plt.ylabel('Humidity')
plt.show()
sns.boxplot(y=data['Pressure Formatted'])
plt.ylabel('Pressure')
plt.show()
#Model training
X=data.drop(columns='temprature formatted')
y=data['temprature formatted']
sscaler = MinMaxScaler() #helps us scale the dataset. This makes it easy for the model to train
cols = X.columns
x_scaled = sscaler.fit_transform(X)
X_scaled = pd.DataFrame(x_scaled, columns = cols)
X_scaled
X_train,X_test, y_train, y_test = train_test_split(X_scaled , y, test_size=0.2,random_state=1)
X_train=np.array(X_train)
X_test=np.array(X_test)
y_train=np.array(y_train)
y_test=np.array(y_test)
y_train
import lightgbm as lightgb
from lightgbm import LGBMRegressor
lgbm=LGBMRegressor()
model_lgb=lgbm.fit(X_train,y_train)
y_pred=model_lgb.predict(X_test)
print('r2 score =',r2_score(y_test,y_pred))

print('Mean absolute error=',mean_absolute_error(y_test,y_pred))
y_test
from pykalman import KalmanFilter
kf = KalmanFilter(transition_matrices=[1],
              observation_matrices = [1],
              initial_state_mean = 0,
              initial_state_covariance = 1,
              observation_covariance=1,
              transition_covariance=.0001)
data['temprature formatted']
mean,cov=kf.filter(data['temprature formatted'])
mean,std=mean.squeeze(),np.std(cov.squeeze())
plt.figure(figsize=(15,7))
plt.plot(data['temprature formatted'] - mean, 'm', lw=1)
plt.plot(np.sqrt(cov.squeeze()), 'y', lw=1)
plt.plot(-np.sqrt(cov.squeeze()), 'c', lw=1)
plt.title('Kalman filter estimate')
plt.legend(['Error: real_value - mean', 'std', '-std'])
plt.xlabel('temperature')
plt.ylabel('Value')