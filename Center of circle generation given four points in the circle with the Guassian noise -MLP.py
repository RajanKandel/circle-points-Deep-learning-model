#!/usr/bin/env python
# coding: utf-8

# Generating the artificial dataset for points in the circle with the guassian noise
# <uo>
#     <li>Each training example consists of four features</li>
#     <li>Dataset consists of 1000 training example and 400 testing example</li>
# <uo>
#     

# In[1]:


#Generate a random points in the circle (with uniform distribution) given the x,y cordinates 
#of the center of circle and radius
import random 
import math
import numpy as np
from numpy.random import normal
class pointInCircle:
    def __init__(self):
        self.r =random.uniform(0,101)
        self.x =random.uniform(-50,50)
        self.y =random.uniform(-50,50)
            
    def randPoint(self):  
        '''generate three points in the circle with uniform distribution'''
        self.r =random.uniform(0,101)
        self.x =random.uniform(-50,50)
        self.y =random.uniform(-50,50)
        points1 = []
        for i in range(0,3):
            #theta
            theta = random.uniform(0, 2*math.pi)
            
            point = (self.x+self.r*math.cos(theta), self.y+self.r*math.sin(theta))
            points1+= point
                 
        return(points1)
    
    def guassianDistPoint(self):
        points2 = normal(loc=0, scale=0.01, size=6)
        return(points2.tolist())
        
    
    def pointsForTraining(self):        
        sum_list = [a + b for a, b in zip(self.randPoint(),self.guassianDistPoint())]
        sum_list.append(self.x)
        sum_list.append(self.y)
        #print(self.r)
        return(sum_list)
    


# In[2]:


h = pointInCircle()
#h.randPoint() 
#h.guassianDistPoint()
#h.pointsForTraining()


# In[3]:


import pandas as pd

train_data_list =[]
for i in range (0,1000):
    train_data_list.append(h.pointsForTraining())
    
test_data_list =[]
for j in range(0,500):
    test_data_list.append(h.pointsForTraining())
    
circles_train_df =pd.DataFrame(train_data_list, columns =['x1','y1','x2','y2','x3','y3','h','k'])
circles_test_df =pd.DataFrame(test_data_list, columns =['x1','y1','x2','y2','x3','y3','h','k'])


# In[4]:


circles_train_df


# In[ ]:


#circles_test_df


# In[5]:


#Convert the dataframe to csv file
circles_train_df.to_csv('circle_trainpoints.csv', index=False)
circles_test_df.to_csv('circle_testpoints.csv', index=False)


# Preparing the data 

# In[6]:


import torch
#print(torch.__version__)


# In[7]:


from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self,trainpoints, testpoints):
        circle_train_df = pd.read_csv(trainpoints)
        circle_test_df = pd.read_csv(testpoints)
        
        self.circle_train_df = circle_train_df.iloc[:,:].values
        self.circle_test_df = circle_test_df.iloc[:,:].values
        
        self.x_circle_train_df = circle_train_df.iloc[:,0:6].values
        self.y_circle_train_df = circle_train_df.iloc[:,6:7].values
        
        self.x_circle_test_df = circle_test_df.iloc[:,0:6].values        
        self.y_circle_test_df = circle_test_df.iloc[:,6:7].values
        
        self.x_tensor_circle_train_df = torch.tensor(self.x_circle_train_df,dtype=torch.float64) 
        self.y_tensor_circle_train_df = torch.tensor(self.y_circle_train_df,dtype=torch.float64) 
        
        self.x_tensor_circle_test_df = torch.tensor(self.x_circle_test_df,dtype=torch.float64) 
        self.y_tensor_circle_test_df = torch.tensor(self.y_circle_test_df,dtype=torch.float64) 
                
           
    def __len__(self):
        return len(self.x_tensor_circle_train_df),len(self.y_tensor_circle_train_df), len(self.x_tensor_circle_test_df), len(self.y_tensor_circle_test_df)
  
    def __getitem__(self,idx):
        return self.x_tensor_circle_train_df[idx],self.y_tensor_circle_train_df[idx]


# In[8]:


p=MyDataset('circle_trainpoints.csv','circle_testpoints.csv')


# In[9]:


#p.circle_train_df


# In[10]:


#p.__getitem__(2)


# In[11]:


from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self,trainpoints):
        circle_train_df = pd.read_csv(trainpoints)
                
        self.circle_train_df = circle_train_df.iloc[:,:].values
                
        self.x_circle_train_df = circle_train_df.iloc[:,0:6].values
        self.y_circle_train_df = circle_train_df.iloc[:,6:7].values
           
        self.x_tensor_circle_train_df = torch.tensor(self.x_circle_train_df,dtype=torch.float64) 
        self.y_tensor_circle_train_df = torch.tensor(self.y_circle_train_df,dtype=torch.float64) 
                       
           
    def __len__(self):
        return len(self.x_tensor_circle_train_df),len(self.y_tensor_circle_train_df)
  
    def __getitem__(self,idx):
        return self.x_tensor_circle_train_df[idx],self.y_tensor_circle_train_df[idx]


# In[12]:


def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # The examples are read at random, in no particular order
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]


# read and print the first small batch of data examples. The shape of the features in each minibatch tells us both the minibatch size and the number of input features. Likewise, our minibatch of labels will have a shape given by batch_size.
# 
# As we run the iteration, we obtain distinct minibatches successively until the entire dataset has been exhausted 

# In[13]:


batch_size = 10
features = p.x_circle_train_df
#features=  torch.from_numpy(features)
labels = p.y_circle_train_df 
#labels = torch.from_numpy(labels)

for X, y in data_iter(batch_size, features, labels):
    print(X, '\n', y)
    break


# In[14]:


#Initializing the model parameter


# initializing weights by sampling random numbers from a normal distribution with mean 0 and a standard deviation of 0.01, and setting the bias to 0.

# In[15]:


w = torch.normal(0, 0.01, size=(6,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)


# updating them until they fit our data sufficiently well. Each update requires taking the gradient of our loss function with respect to the parameters. Given this gradient, we can update each parameter in the direction that may reduce the loss.
# 

# In[47]:


def linreg(X, w, b):  #@save
    return torch.matmul(X, w) + b


# Defining the loss function

# In[48]:


def squared_loss(y_hat, y):  
    """Squared loss."""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


# Optimizing Algorithm

# In[49]:


def sgd(params, lr, batch_size):  
    """Minibatch stochastic gradient descent."""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


# In[51]:


#Training
#automatic differentiation, to compute the gradient.
lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)  # Minibatch loss in `X` and `y`
        # gradient on `l` with respect to [`w`, `b`]
        l.sum().backward()
        sgd([w, b], lr, batch_size)  # Update parameters using their gradient
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')


# In[ ]:




