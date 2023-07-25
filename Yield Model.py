import numpy as np
import pandas as pd
import tensorflow as tn
from sklearn.preprocessing import LabelEncoder
dataset=pd.read_csv('/Users/divyanshyadav/Downloads/dataset.csv')
print(dataset.isnull().sum())
# dataset['water'] = dataset['water'].replace(np.NaN, dataset['water'].mean())
# dataset['uv'] = dataset['uv'].replace(np.NaN, dataset['uv'].mean())
dataset=dataset.dropna(axis=0)
print(dataset.isnull().sum())
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values
print(x)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x=sc.fit_transform(x)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train,y_test=train_test_split(x,y,test_size=.2, random_state=25)
ann=tn.keras.models.Sequential()
ann.add(tn.keras.layers.Dense(units=36,activation='elu'))
ann.add(tn.keras.layers.Dense(units=36,activation='elu'))
ann.add(tn.keras.layers.Dense(units=36,activation='elu'))
ann.add(tn.keras.layers.Dense(units=36,activation='elu'))
ann.add(tn.keras.layers.Dense(units=36,activation='elu'))
ann.add(tn.keras.layers.Dense(units=36,activation='elu'))
ann.add(tn.keras.layers.Dense(units=36,activation='elu'))
ann.add(tn.keras.layers.Dense(units=36,activation='elu'))
ann.add(tn.keras.layers.Dense(units=36,activation='elu'))
ann.add(tn.keras.layers.Dense(units=1))
ann.compile(optimizer='adam',loss='mean_squared_error')
hist=ann.fit(x_train,y_train,validation_data=(x_test,y_test),batch_size=10,epochs =200)
y_pred=ann.predict(x_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))
from sklearn.metrics import r2_score
print(r2_score(y_test,y_pred))
import matplotlib.pyplot as plt
plt.plot(hist.history['loss'],color='red')
plt.plot(hist.history['val_loss'],color='blue')
plt.legend(["training loss", "validation loss"])
plt.show()
plt.plot(y_test,color='red')
plt.plot(y_pred,color='blue')
plt.xlim(0, 70)
plt.ylim(0,100)
plt.show()
from sklearn.metrics import r2_score
r2_score(y_test,y_pred)