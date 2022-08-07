#!/usr/bin/env python
# coding: utf-8

# In[63]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score


# In[64]:


parkinsons_data = pd.read_csv('parkinsons.csv')


# In[65]:


type(parkinsons_data)


# In[66]:


parkinsons_data.head(4)


# In[7]:


parkinsons_data.shape


# In[9]:


parkinsons_data.info()


# In[67]:


parkinsons_data.isnull().sum()


# In[12]:


parkinsons_data.describe()


# In[68]:


parkinsons_data['status'].value_counts()


# In[69]:


parkinsons_data.groupby('status').mean()


# In[70]:


X = parkinsons_data.drop(columns=['name','status'],axis=1)
Y = parkinsons_data['status']


# In[21]:


print(Y)


# In[71]:


X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size= 0.2, random_state=2)


# In[72]:


print(X.shape,X_train.shape,X_test.shape)


# In[73]:


scaler = StandardScaler()


# In[74]:


scaler.fit(X_train)


# In[75]:


X_train = scaler.transform(X_train)
X_test =scaler.transform(X_test)


# In[33]:


print(X_train)


# In[76]:


model = svm.SVC(kernel = 'linear')


# In[77]:


model.fit(X_train,Y_train)


# In[78]:


X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train,X_train_prediction)


# In[79]:


print('Accuracy Score of training data:' ,training_data_accuracy)


# In[80]:


X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(Y_test , X_test_prediction)


# In[81]:


print('Accuracy Score of test data:' ,test_data_accuracy)


# In[93]:


input_data = (197.07600,206.89600,192.05500,0.00289,0.00001,0.00166,0.00168,0.00498,0.01098,0.09700,0.00563,0.00680,0.00802,0.01689,0.00339,26.77500,0.422229,0.741367,-7.348300,0.177551,1.743867,0.085569)


# In[99]:


input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
st_data = scaler.transform(input_data_reshaped)
prediction = model.predict(st_data)
if (prediction == 0):
    print('The person does not have Parkinsons disease.')
else: 
    print('The person has Parkinsons disease')

