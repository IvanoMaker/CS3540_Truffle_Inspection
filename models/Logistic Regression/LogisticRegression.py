import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split




imagesPath = "C:/Users/boyle/PycharmProjects/TruffleDefectDetection/x_data_flattened.npy"
labelsPath = "C:/Users/boyle/PycharmProjects/TruffleDefectDetection/y_labels.npy"


images = np.load(imagesPath) #pictures of the truffles flattened
labels = np.load(labelsPath) #lablels for the pictures

normalizedImages = images / 255

ones = np.ones((normalizedImages.shape[0],1)) # bias
X = np.hstack([ones, normalizedImages])  #stack bias to left of image numbers
Y = labels.reshape(-1,1)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.2, random_state = 42, stratify = Y)

def hypothesis(X, theta):
    # h =  write your code here.
    h = 1/(1+np.exp((-np.dot(X,theta))))
    return h

def calcLogRegressionCost(X, Y, theta):
    m = len(X)

    h = hypothesis(X, theta)

    cost = (-1/m)*(np.dot(Y.T,np.log(h))+np.dot((1-Y).T,np.log(1-h)))
    return cost[0, 0]



def logRegressionGradientDescent(X, Y, theta, eta, iters):
    m = len(X)
    h = 0
    gradient = 0
    cost = np.zeros(iters)
    for i in range(iters):
        h = hypothesis(X, theta)
        gradient = (2/(m)) * np.dot(X.T, (h - Y))
        theta = theta - eta * gradient
        cost[i] = calcLogRegressionCost(X, Y, theta)
    return theta, cost


eta = 0.1
iters = 3000
theta = np.zeros((X_train.shape[1], 1))
print('Training')
theta, cost = logRegressionGradientDescent(X_train, Y_train, theta, eta, iters)
print('Final Cost:')
print(cost[-1])


plt.figure()
sns.lineplot(x = range(iters), y = cost)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.show()
