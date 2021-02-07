#!/usr/bin/env python
# coding: utf-8

# In[4]:



import pandas as pd


# In[5]:


get_ipython().system('pip install pandas')


# In[6]:


import pandas as pd


# In[7]:


df=pd.read_csv('car data.csv')


# In[8]:


df.head()


# In[9]:


df.shape


# In[10]:


print(df['Seller_Type'].unique())
print(df['Transmission'].unique())
print(df['Owner'].unique())


# In[11]:


## check missing values
df.isnull().sum()


# In[12]:


df.describe()


# In[13]:


df.columns


# In[14]:


final_dataset=df[[ 'Year', 'Selling_Price', 'Present_Price', 'Kms_Driven',
       'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner']]


# In[15]:


final_dataset.head()


# In[16]:


final_dataset['Current_Year']=2020


# In[17]:


final_dataset.head()


# In[18]:


final_dataset['no_year']=final_dataset['Current_Year']-final_dataset['Year']


# In[19]:


final_dataset.head()


# In[20]:


final_dataset.drop(['Year'],axis=1,inplace=True)


# In[21]:


final_dataset.drop(['Current_Year'],axis=1,inplace=True)


# In[22]:


final_dataset.head()


# In[23]:


final_dataset=pd.get_dummies(final_dataset,drop_first=True)


# In[24]:


final_dataset


# In[25]:


final_dataset


# In[26]:


final_dataset.corr()


# In[27]:


get_ipython().system('pip install seaborn')


# In[93]:


import seaborn as sns


# In[29]:


sns.pairplot(final_dataset)


# In[92]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[38]:


corrmat=final_dataset.corr()


# In[35]:





# In[48]:


plt.figure((figsize)=(20,20))
g=sns.heatmap(final_dataset.corr(),annot=True,cmap='RdYlGn')


# In[ ]:





# In[ ]:





# In[46]:


corrmat=final_dataset.corr()
#top_corr_features=corrmat.index()
plt.figure((figsize)=(20,20))
#plot heat map
g=sns.heatmap(final_dataset[top_corr_features].corr(),annot=True,cmap='RdYlGn')


# In[50]:


final_dataset.head()


# In[52]:


#define dependent and independent features
X=final_dataset.iloc[:,1:]
y=final_dataset.iloc[:,0]


# In[54]:


X


# In[55]:


y


# In[60]:


#feature importance
from sklearn.ensemble import ExtraTreesRegressor
model=ExtraTreesRegressor()
model.fit(X,y)


# In[59]:


print(model.feature_importances_)


# In[61]:


feat_importances = pd.Series(model.feature_importances_, index=X.columns)


# In[62]:


feat_importances


# In[63]:


feat_importances.nlargest(5)


# In[69]:


feat_importances.nlargest(5).plot(kind='barh')
plt.show()


# In[70]:


# create train/test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[71]:


X_train


# In[72]:


X_test


# In[74]:


from sklearn.ensemble import RandomForestRegressor
rf_random= RandomForestRegressor()


# In[78]:


##Hyperparameters
import numpy as np

n_estimators=[int(x) for x in np.linspace(start=100, stop=1200, num=12)]
n_estimators


# In[82]:


# Randomized Search CV
max_features=['auto','sqrt']
max_depth=[int(x) for x in np.linspace(5,30,num=6)]
min_samples_split = [2,5,10,15,100]
min_samples_leaf = [1,2,5,10]


# In[80]:


from sklearn.model_selection import RandomizedSearchCV


# In[ ]:


######create the random grid


# In[83]:


random_grid = {'n_estimators':n_estimators,
               'max_features':max_features,
               'max_depth':max_depth,
               'min_samples_split':min_samples_split,
               'min_samples_leaf':min_samples_leaf
}

print(random_grid)


# In[84]:


#search for best hyperparameters
rf = RandomForestRegressor()


# In[88]:


rf_random = RandomizedSearchCV (estimator = rf, param_distributions = random_grid, scoring = 'neg_mean_squared_error',n_iter=10, cv=5,verbose=2,random_state=42,n_jobs=1)


# In[89]:


rf_random


# In[90]:


rf_random.fit(X_train,y_train)


# In[100]:


predictions=rf_random.predict(X_test)

#caldulate the difference between y_test and prediction

sns.histplot(y_test-predictions)


# In[101]:


plt.scatter(y_test,predictions)


# In[103]:


import pickle

# open a file, where you want to store the data

file=open('random_forest_regression_model.pk1','wb')

#dump information to that file

pickle.dump(rf_random,file)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:
