import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
import pickle
from sklearn.impute import SimpleImputer


house_data = pd.read_csv("realest.csv")

# house_data.head()
# house_data.shape

# house_data.describe()
# house_data.tail()
# house_data.isnull().sum()

house_data = house_data.iloc[0:156,0:9]

SIN = SimpleImputer(missing_values=np.nan, strategy='mean')
house_data.iloc[0:156,0:6]=SIN.fit_transform(house_data.iloc[0:156,0:6])
house_data.iloc[0:156,0:6]

SIC = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
house_data.iloc[0:156,6:9]=SIC.fit_transform(house_data.iloc[0:156,6:9])
house_data.iloc[0:156,6:9]

house_data = pd.merge(house_data.iloc[0:156,0:6],house_data.iloc[0:156,6:9],right_index=True,left_index=True)

house_data.iloc[:,1:].corrwith(house_data.iloc[:,0],axis=0)

# sns.heatmap(house_data.corr(),annot=True)

x_train,x_test,y_train,y_test=train_test_split(house_data.iloc[:,1:],house_data.iloc[:,0],train_size=0.8,random_state=1)
LR=LinearRegression()
LR.fit(x_train,y_train)

y_pred=LR.predict(x_test)
y_pred
LR.score(x_train,y_train)

print(r2_score(y_test,y_pred))
print(mean_squared_error(y_pred,y_test))

y_pred=GradientBoostingRegressor().fit(x_train,y_train).predict(x_test)
print(mean_squared_error(y_test,y_pred))
print(r2_score(y_test,y_pred))
print(GradientBoostingRegressor().fit(x_train,y_train).score(x_train,y_train))

pickle.dump(LR,open('model.pkl','wb'))
