#!/usr/bin/env python
# coding: utf-8

# # Data Science Project : Diabetes prediction using Machine Learning

# ## 1.Data collection

# In[2]:


#importing required modules
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


#reading the dataset
df = pd.read_csv("./diabetes.csv")


# ## 2. Data Exploration and Preprocessing

# In[4]:


#dimensions
df.shape


# In[5]:


df.head(5)


# In[6]:


#nullvalues
df.isnull().values.any()


# In[7]:


#no null values


# In[8]:


#removing duplicates
df.drop_duplicates(inplace=True)


# In[9]:


df.shape


# In[10]:


#no change in rows hence no duplicates found


# In[11]:


df.info()


# In[12]:


df.describe()


# In[13]:


# here we can see that min for some attributes(bloodpressure,glucose,skin thickness) is zero fill them with mean


# In[14]:


df.fillna(0,inplace=True)#there are no null values but if any then it fills them with 0


# In[15]:


# Now we replace all the 0s with respective attribute means


# In[16]:


df.replace({'BloodPressure':0},df['BloodPressure'].mean(),inplace=True)
df.replace({'Glucose':0},df['Glucose'].mean(),inplace=True)
df.replace({'SkinThickness':0},df['SkinThickness'].mean(),inplace=True)
df.replace({'Insulin':0},df['Insulin'].mean(),inplace=True)
df.replace({'BMI':0},df['BMI'].mean(),inplace=True)


# In[17]:


df.describe()


# In[18]:


#now we can see that the min for attributes is not zero


# ## 3. Data Visualization

# In[23]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[24]:


plt.figure(figsize=(8,8))
pieC=df['Outcome'].value_counts()
explode=(0.05,0)
colors=['green','red']
labels=['0 - Non Diabetic','1 - Diabetic']
sns.set(font_scale=1.5)
plt.pie(pieC,labels=('0 - Non Diabetic','1 - Diabetic'),autopct="%.2f%%",explode=explode,colors=colors)
plt.legend(labels,loc='lower left')


# In[25]:


# There are 65.10% non-diabetic points and 34.90% diabetic points


# In[26]:


# Histogram


# In[27]:


df.hist(bins=10,figsize=(10,10))
plt.show()


# In[ ]:


# Scatterplot


# In[28]:


from pandas.plotting import scatter_matrix
scatter_matrix(df,figsize=(20,20))


# In[ ]:


# Pairplot


# In[29]:


sns.pairplot(data=df,hue='Outcome')
plt.show()


# In[30]:


#correlation


# In[33]:


corrmat=df.corr()
top_corr_features=corrmat.index
plt.figure(figsize=(10,10))
g=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")


# In[34]:


# Glucose is having highest correlation with outcome among all.


# ### Splitting of data

# In[36]:


target_name='Outcome'
y=df[target_name]
x=df.drop(target_name,axis=1)


# In[37]:


x.head()


# In[38]:


y.head()


# In[ ]:


# Feature Scaling


# In[40]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(x)
SSX=scaler.transform(x)


# ### Train and Test Dataset

# In[41]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(SSX,y,test_size=0.2,random_state=7)


# In[42]:


x_train.shape,y_train.shape


# In[43]:


x_test.shape,y_test.shape


# ### Building the model with different classification algorithms

# ### K Nearest Neighbors Algorithm

# In[44]:


from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()
knn.fit(x_train,y_train)


# In[46]:


knn_pred=knn.predict(x_test)


# In[63]:


# Accuracy Prediction
from sklearn.metrics import accuracy_score
print("Train Accuracy:",knn.score(x_train,y_train)*100)
print("Accuracy(Test) score:",accuracy_score(y_test,knn_pred)*100)


# In[48]:


# Confusion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,knn_pred)


# In[49]:


#  74.67% accuracy with KNearest Neighbors Algorithm


# ### Naive Bayes Algorithm

# In[50]:


from sklearn.naive_bayes import GaussianNB
nb=GaussianNB()
nb.fit(x_train,y_train)


# In[51]:


nb_pred=nb.predict(x_test)


# In[62]:


# Accuracy Prediction
from sklearn.metrics import accuracy_score
print("Train Accuracy:",nb.score(x_train,y_train)*100)
print("Accuracy(Test) score ",accuracy_score(y_test,nb_pred)*100)


# In[53]:


# Confusion Matrix
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,nb_pred)


# In[55]:


# 74.02% accuracy with Naive Bayes Algorithm.


# ### Logistic Regression

# In[56]:


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver='liblinear',multi_class='ovr')
lr.fit(x_train,y_train)


# In[57]:


lr_pred=lr.predict(x_test)


# In[60]:


# Accuracy Prediction
from sklearn.metrics import accuracy_score
print("Train Accuracy:",lr.score(x_train,y_train)*100)
print("Accuracy(Test) score:",accuracy_score(y_test,lr_pred)*100)


# In[59]:


# Confusion Matrix
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,lr_pred)


# In[61]:


#  77.27% accuracy with Logistic Regression.


# ### Decision Tree Algorithm

# In[66]:


from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
dt.fit(x_train,y_train)


# In[67]:


dt_pred=dt.predict(x_test)


# In[86]:


# Accuracy Prediction
from sklearn.metrics import accuracy_score
print("Train Accuracy: ",dt.score(x_train,y_train)*100)
print("Accuracy(Test) score:",accuracy_score(y_test,dt_pred)*100)


# In[70]:


# Confusion Matrix
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,dt_pred)


# In[72]:


# 79.22% accuracy with Decision Tree  Classifier


# ### Random Forest  Algorithm

# In[73]:


from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(criterion='entropy')
rf.fit(x_train,y_train)


# In[74]:


rf_pred=rf.predict(x_test)


# In[89]:


# Accuracy Prediction
from sklearn.metrics import accuracy_score
print("Train Accuracy: ",rf.score(x_train,y_train)*100)
print("Accuracy(Test) score: ",rf.score(x_test,y_test)*100)


# In[78]:


# Confusion Matrix
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,rf_pred)


# In[79]:


# 80.51% accuracy from Random Forest Classifier.


# ### Support Vector Machine Algorithm

# In[80]:


from sklearn.svm import SVC
sv=SVC()
sv.fit(x_train,y_train)


# In[81]:


sv_pred=sv.predict(x_test)


# In[88]:


# Accuracy Prediction
from sklearn.metrics import accuracy_score
print("Train Accuracy of Support Vector Machine",sv.score(x_train,y_train)*100)
print("Accuracy(Test) score:",accuracy_score(y_test,sv_pred)*100)


# In[84]:


# Confusion Matrix
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,sv_pred)


# In[87]:


#  83.11% accuracy with Support Vector Machine


# ### Result :  We got highest accuracy of 83.11% using the Support Vector Machine Algorithm in classifying the datapoints as diabetic and non-diabetic
