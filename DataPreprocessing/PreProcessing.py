#%%

import numpy as np
import pandas as pd

#%%

dataset=pd.read_csv("D:\Git\ML_Projects\DataPreprocessing\Data.csv")
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,3].values

#%%
#Handling Missing Data

from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=np.nan, strategy='mean')
imputer=imputer.fit(X[:,1:3])
X[:,1:3]=imputer.transform(X[:,1:3])

#%%
#Including Categorical Data

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_X=LabelEncoder()
X[:,0]=labelencoder_X.fit_transform(X[:,0])

onehotencoder=OneHotEncoder(categorical_features=[0])
X=onehotencoder.fit_transform(X).toarray()
#onehotencoder=ColumnTransformer([("Country", OneHotEncoder(), [1])], remainder = 'passthrough')
#X=onehotencoder.fit_transform(X).toarray()

#%%

labelencoer_Y=LabelEncoder()
Y=labelencoer_Y.fit_transform(Y)

#%%
#Splitting Dataset into Training and Testing 

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test=train_test_split(X, Y, test_size=0.25,random_state=42)

#%%
#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.fit_transform(X_test)

#%%