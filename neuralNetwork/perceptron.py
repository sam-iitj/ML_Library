import numpy as np 

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
      print compute
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

X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])  
X_ = np.column_stack((np.ones(X.shape[0]), X))
y = np.array([-1, 1, 1, 1])
p = Perceptron(0.01, 100)    
p.gradientDescent(X_, y)
print p.weights
print p.perceptronOutput(X_)
