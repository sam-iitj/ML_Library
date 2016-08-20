import numpy as np 
import matplotlib.pyplot as plt

class Perceptron:

  def __init__(self, alpha, iterations):
    self. weights = None
    self.alpha = alpha 
    self.iterations = iterations
  
  def perceptronOutput(self, X):
    compute = np.dot(X, self.weights)
    if isinstance(compute, np.float64):
      if compute > 0:
        return 1
      else:
        return -1
    else:
      final = [] 
      for elem in compute:
        if elem > 0.0:
          final.append(1)
        else:
          final.append(0)
      return final 
       
  def gradientDescent(self, X_, y):
    self.weights = np.random.rand(X_.shape[1])
    for iter_ in range(self.iterations):
      delta_weights = self.weights
      for i in range(X_.shape[0]):
        o = self.perceptronOutput(X_[i, :])
        for j in range(self.weights.shape[0]):
          delta_weights[j] += self.alpha * (y[i] - o) * X_[i, j]
      self.weights = delta_weights 

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  
X_ = np.column_stack((np.ones(X.shape[0]), X))
y = np.array([-1, -1, 1, -1])
rates = [1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1, 5, 10, 20]

for rate in rates:
  print "Current Learning Rate : "  + str(rate)
  p = Perceptron(rate, 1000)    
  p.gradientDescent(X_, y)
  print p.weights
  print p.perceptronOutput(X_)
  for i in range(len(y)):
    if y[i] == 1:
      plt.scatter(X_[i, 1], X_[i, 2], color = "g")
    else:
      plt.scatter(X_[i, 1], X_[i, 2], color = "c")

  x1 = np.random.randint(low=-5, high=5, size=100)
  x2 = [ (-p.weights[0] - p.weights[1] * elem)/p.weights[2] for elem in x1]
  plt.plot(x1, x2, color = "b")
  plt.show()
