import numpy as np 

class LinearRegression:
          def __init__(self, data, labels, learning_rate):
              self.model = None
              data = (data - np.mean(data,  axis=0))/np.std(data, axis=0)
              self.X = np.vstack((np.ones(data.shape[1]).T, np.array(data))).T
              self.labels = labels
              self.theta = np.random.rand(self.X.shape[1])
              self.M = self.X.shape[0]
              self.D = self.X.shape[1]
              self.alpha = learning_rate
     
          def gradient_descent(self):
              self.theta -= (self.alpha/self.M)* (self.hypothesis() - self.labels).dot(self.X)
     
          def cost(self):
              return np.sum(np.power(self.hypothesis() - self.labels, 2))/self.M
     
          def hypothesis(self):
              return np.dot(self.X, self.theta)

data = np.array([[1, 2, 3, 4], [4, 5, 1, 2]])
labels = np.array([1, 2, 3, 4])
lr = LinearRegression(data, labels, 0.01)

for i in range(10):
    print lr.cost()
    lr.gradient_descent()
