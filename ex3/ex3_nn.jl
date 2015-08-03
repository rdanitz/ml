using MAT

include("predict.jl")
include("displayData.jl")

function run()
  data = matread("ex3data1.mat")
  weights = matread("ex3weights.mat")

  X = data["X"]
  y = data["y"]
  Θ1 = weights["Theta1"]
  Θ2 = weights["Theta2"]

  predict(Θ1, Θ2, X)
  
  m = size(X, 1)
  for i in randperm(m)
    println("Displaying Example Image")
    displayData(X[i,:])
    p = predict(Θ1, Θ2, X[i,:])
    println("Neural Network Prediction: ", p[1] % 10)
    readline()
  end
end
