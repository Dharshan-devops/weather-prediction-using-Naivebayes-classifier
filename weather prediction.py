#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df=pd.read_csv("tennis.csv")
df


# In[4]:


X_train = pd.get_dummies(df[['outlook', 'temp', 'humidity', 'windy']])
y_train = pd.DataFrame(df['play'])


# In[5]:


print(X_train.info())
print(X_train.head())


# In[7]:


print(y_train.info())


# In[8]:


print(y_train)


# In[9]:


from sklearn.naive_bayes import GaussianNB


# In[10]:


classifier=GaussianNB()


# In[12]:


classifier.fit(X_train,y_train)


# In[13]:


classifier.score(X_train,y_train)


# In[14]:


X_train.head()


# In[15]:


classifier.predict([[True,0,0,1,0,1,0,1,0]])


# In[16]:


y_train.head()


# In[19]:


a=classifier.predict([[True,0,0,1,0,1,1,1,1]])


# In[20]:


if(a[0]=="yes"):
    print("yOU CAN PLAY!!!!")
else:
    print("You cant play!!!")
    


# In[22]:


pip install pillow


# In[23]:


import PIL


# In[25]:


from PIL import Image


# In[ ]:


im=Image.open()

