sigmoid(z) = 1./(1+exp(-z))
sigmoidGradient(z) = sigmoid(z).*(1-sigmoid(z))
