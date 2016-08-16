import numpy as np 
from numpy.linalg import inv
import math
import pylab
from scipy import stats
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt

def hypothesis(x, theta):
  return np.dot(x, theta)

def cost(m, X, y, theta, k, alpha):
  """
  Cost function for ridge regression. 
  """
  cost_ = 0.0
  for i in range(m):
    hypo = hypothesis(X[i, :], theta)
    cost_ += (y[i] - hypo)*X[i, k] + alpha*theta[k]
  return cost_

def gradient_descent(m, X, y, theta, alpha, iterations):
  X = np.column_stack((np.ones(X.shape[0]), X))
  iter_ = 0 
  while iter_ < iterations:
    new_theta = theta 
    for i in range(theta.shape[0]):
      new_theta[i] += alpha * cost(m, X, y, theta, i, alpha)
    theta = new_theta
    iter_ += 1
  return theta

def error(X, y, theta):
  y_predicted = np.dot(np.column_stack((np.ones(X.shape[0]), X)), theta)
  error = sum([math.pow(y[i] - y_predicted[i], 2) for i in range(y_predicted.shape[0])])
  return error/2.0

X = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
y = np.array([math.pow(elem, 3) for elem in X])

# Additing x^2 feature to data
X_ = np.column_stack((X, np.array([math.pow(elem, 2) for elem in X])))
alpha_ridge = [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2]
min_ = 1000000000000
final_alpha = 0.0
iterations = 1000

for elem in alpha_ridge:
  try:
    theta = np.array([0.1, 0.1, 0.1])
    theta = gradient_descent(X.shape[0], X_, y, theta, elem, iterations)
    if error(X_, y, theta) != np.nan and error(X_, y, theta) < min_:
      min_ = error(X_, y, theta)
      final_alpha = elem
      print elem,theta, error(X_, y, theta)
  except:
    pass

print "Final alpha :" + str(final_alpha)
theta = np.array([0.1, 0.1, 0.1])
theta = gradient_descent(X.shape[0], X_, y, theta, final_alpha, iterations)
y_predict = np.dot(np.column_stack((np.ones(X.shape[0]), X_)), theta)
print theta

from sklearn.linear_model import Ridge
clf = Ridge(alpha=0.0001)
clf.fit(np.column_stack((np.ones(X.shape[0]), X_)), y)
print clf.coef_
print clf.intercept_
print clf

plt.scatter(X, y)
plt.scatter(X, y_predict, color = "r")
plt.show()
