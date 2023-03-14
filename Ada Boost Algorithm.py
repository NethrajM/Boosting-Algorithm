#!/usr/bin/env python
# coding: utf-8

# In[39]:


import pandas as pd 


# In[40]:


Dataset=pd.read_csv("insurance_pre.csv")


# In[41]:


Dataset


# In[42]:


Dataset=pd.get_dummies(Dataset,drop_first=True)


# In[43]:


Dataset


# In[44]:


Dataset.columns 


# In[45]:


independent=Dataset[['age', 'bmi', 'children','sex_male', 'smoker_yes']]


# In[46]:


independent


# In[47]:


Dependent=Dataset[['charges']]


# In[48]:


Dependent


# In[49]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(independent,Dependent,test_size=50,random_state=0)


# In[55]:


# Importing necessary libraries
from sklearn.ensemble import AdaBoostRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
# Generating a random regression dataset
X, y = make_regression(n_samples=1000, n_features=10, random_state=42)
# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating an instance of Adaboost Regressor
adaboost_reg = AdaBoostRegressor(random_state=100, learning_rate=1)
# Fitting the model

adaboost_reg.fit(X_train, y_train)


# In[52]:


y_pred = adaboost_reg.predict(X_test)


# In[53]:


mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)


# In[ ]:





# In[ ]:




