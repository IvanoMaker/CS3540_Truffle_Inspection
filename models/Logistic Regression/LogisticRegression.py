import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import seaborn as sns


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
    h = np.clip(1/(1+np.exp((-np.dot(X,theta)))), 1e-10, 1-1e-10)
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


eta = 0.0001
iters = 5000
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

print('Gona Predict')
def predict(X, theta, threshold=0.5):
    return (hypothesis(X, theta) >= threshold).astype(int)

Y_predict = predict(X_test, theta)
print('checking accuracy')
accuracy = np.mean(Y_predict == Y_test)

print(f"Accuracy = {accuracy}")

TrueDefective = np.sum((Y_predict == 1) & (Y_test == 1))
TrueSatisfactory = np.sum((Y_predict == 0) & (Y_test == 0))
FalseDefective = np.sum((Y_predict == 1) & (Y_test == 0))
FalseSatisfactory = np.sum((Y_predict == 0) & (Y_test == 1))

print(f"True Defective Count: {TrueDefective}")
print(f"True Satisfactory Count: {TrueSatisfactory}")
print(f"False Defective Count: {FalseDefective}")
print(f"False Satisfactory Count: {FalseSatisfactory}")

precision = TrueDefective / (TrueDefective + FalseDefective)
print(f"Precision = {precision}")