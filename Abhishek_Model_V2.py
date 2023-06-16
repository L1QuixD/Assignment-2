#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# In[2]:


df = pd.read_csv("data.csv")
df.head()


# In[3]:


df.info()


# In[4]:


df.drop(["id", "Unnamed: 32"], axis = 1, inplace = True)


# In[5]:


df.describe().T


# In[6]:


df.diagnosis = [1 if each == "M" else 0 for each in df.diagnosis]


# In[7]:


df.info()


# In[8]:


y = df.diagnosis.values.reshape(-1, 1)
X = df.iloc[:, 1:].values


# In[9]:


X = ((X - np.min(X))/(np.max(X)-np.min(X)))


# In[10]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# In[11]:


X_train, X_test, y_train, y_test = X_train.T, X_test.T, y_train.T, y_test.T


# In[12]:


print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# In[13]:


def initialize_weights_and_bias(dimension):
    w = np.full((dimension, 1), 0.01)
    b = 0.0
    return w, b


# In[14]:


def sigmoid(z):
    y_head = (1 / (1 + np.exp(-z)))
    return y_head


# In[15]:


def forward_backward_propagation(w,b,X_train,y_train):
    # forward propagation
    z = np.dot(w.T,X_train) + b
    y_head = sigmoid(z)
    loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)
    cost = (np.sum(loss))/X_train.shape[1]      # X_train.shape[1]  is for scaling
    
    # backward propagation
    derivative_weight = (np.dot(X_train,((y_head-y_train).T)))/X_train.shape[1] # X_train.shape[1]  is for scaling
    derivative_bias = np.sum(y_head-y_train)/X_train.shape[1]                 # X_train.shape[1]  is for scaling
    gradients = {"derivative_weight": derivative_weight, "derivative_bias": derivative_bias}
    
    return cost,gradients


# In[20]:


# Updating(learning) parameters
def update(w, b, X_train, y_train, learning_rate,number_of_iterarion):
    cost_list = []
    cost_list2 = []
    index = []
    
    # updating(learning) parameters is number_of_iterarion times
    for i in range(number_of_iterarion):
        # make forward and backward propagation and find cost and gradients
        cost,gradients = forward_backward_propagation(w,b,X_train,y_train)
        cost_list.append(cost)
        # lets update
        w = w - learning_rate * gradients["derivative_weight"]
        b = b - learning_rate * gradients["derivative_bias"]
        if i % 10 == 0:
            cost_list2.append(cost)
            index.append(i)
            print ("Cost after iteration %i: %f" %(i, cost))
            
            
            
    # we update(learn) parameters weights and bias
    parameters = {"weight": w,"bias": b}
    plt.plot(index,cost_list2)
    plt.xticks(index,rotation='vertical')
    plt.xlabel("Number of Iterarion")
    plt.ylabel("Cost")
    plt.show()
    return parameters, gradients, cost_list


# In[21]:


#  # prediction
def predict(w,b,X_test):
    # X_test is a input for forward propagation
    z = sigmoid(np.dot(w.T,X_test)+b)
    Y_prediction = np.zeros((1,X_test.shape[1]))
    # if z is bigger than 0.5, our prediction is sign one (y_head=1),
    # if z is smaller than 0.5, our prediction is sign zero (y_head=0),
    for i in range(z.shape[1]):
        if z[0,i]<= 0.5:
            Y_prediction[0,i] = 0
        else:
            Y_prediction[0,i] = 1

    return Y_prediction


# In[23]:


def logistic_regression(X_train, y_train, X_test, y_test, learning_rate ,  num_iterations):
    # initialize
    dimension =  X_train.shape[0]  # that is 30
    w,b = initialize_weights_and_bias(dimension)
    # do not change learning rate
    parameters, gradients, cost_list = update(w, b, X_train, y_train, learning_rate,num_iterations)
    
    y_prediction_test = predict(parameters["weight"],parameters["bias"],X_test)

    # Print test Errors
    print(f"test accuracy: % {(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100)} ")
    
logistic_regression(X_train, y_train, X_test, y_test,learning_rate = 1, num_iterations = 300) 


# In[25]:


# sklearn with LR
lr = LogisticRegression()
lr.fit(X_train.T,y_train.T)
print(f"Test Accuracy {lr.score(X_test.T,y_test.T) * 100}")


# In[ ]:




