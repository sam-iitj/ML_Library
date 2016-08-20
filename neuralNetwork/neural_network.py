import numpy as np 

class NeuralNetwork:

  def __init__(self, input_layers, hidden_layers, output_layers, learning_rate, iterations):
    self.input_layers = input_layers
    self.hidden_layers = hidden_layers
    self.output_layers = output_layers
    self.learning_rate = learning_rate
    self.iterations = iterations
    self.weights_input_hidden = np.random.rand(self.hidden_layers, self.input_layers)
    self.weights_hidden_output = np.random.rand(self.output_layers, self.hidden_layers)

  def sigmoid(self, theta, X):
    return 1.0/(1.0 + np.exp(np.dot(X, theta)))

  def fit(self, X, y):
    # Values propogated to hidden layers in the network.
    for i in range(X.shape[0]):
      elem = X[i, :]

      # Calculate the output of the hidden layers using the current weights between the input layer and hidden layer
      hidden_output = []
      for h in range(self.hidden_layers):
        hidden_output.append(self.sigmoid(self.weights_input_hidden[h, :], elem))

      # Calculate the output of the output layer using the current weight between the hidden layer and the output layer
      output_values = []
      for o in range(self.output_layers):
        output_values.append(self.sigmoid(self.weights_hidden_output[o, :], hidden_output))

      # Step 2 - Let's compute the error for each output unit 
      output_unit_error = []
      for k in range(self.output_layers):     
        output_unit_error.append(output_values[k] * ( 1 - output_values[k]) * (y[i][k] - output_values[k]))

      # Step 3 - Let's compute the error in hidden layer unit 
      hidden_layer_error = []
      for h in range(self.hidden_layers):
        temp = np.dot( self.weights_hidden_output[h, :] , output_unit_error)
        hidden_layer_error.append(hidden_output[h] * ( 1 - hidden_output[h] ) * temp)

      # Step 4 Let's update all the weights 
      for i in range(self.input_layers):
        for j in range(self.hidden_layers):
          self.weights_input_hidden[j, i] += self.learning_rate * hidden_layer_error[j] * X[j, i]  

      for i in range(self.hidden_layers):
        for j in range(self.output_layers):
          self.weights_hidden_output[i, j] += self.learning_rate * output_unit_error[j] * hidden_output[i]     

  def train(self, X, y):
    for iter_ in range(self.iterations):
      self.fit(X, y)

  def predict(self, X):
    print("Current Input : " + str(X))
    hidden_output = []
    for h in range(self.hidden_layers):
      hidden_output.append(self.sigmoid(self.weights_input_hidden[h, :], X))

    output = []
    for o in range(self.output_layers):
      output.append(self.sigmoid(self.weights_hidden_output[o, :], hidden_output))

    return output

nn = NeuralNetwork(3, 2, 2, 0.01, 1000)
X = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
y = np.array([[0, 0], [0, 0], [0, 0], [0, 1], [0, 0], [0, 1], [1, 0], [1, 1]])
#X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
#y = np.array([[0,0], [0,  1], [1, 0], [0, 0]])
nn.train(X, y)
print(nn.weights_hidden_output)
print(nn.weights_input_hidden)
print("\n\n Prediction Phase \n\n")
for i in range(X.shape[0]):
  print(nn.predict(X[i, :]))
  print("Actual Output : " + str(y[i]))
