using MAT

include("sigmoidGradient.jl")
include("nnCostFunction.jl")
include("checkNNGradients.jl")
include("trainNN.jl")
include("predict.jl")

function run()
  data = matread("ex4data1.mat")
  weights = matread("ex4weights.mat")

  X = data["X"]
  y = data["y"]
  Θ1 = weights["Theta1"]
  Θ2 = weights["Theta2"]

  λ = 1
  s = {400, 25, 10}
  θ = [Θ1[:], Θ2[:]]

  J, _ = nnCostFunction(θ, s, X, y)
  println("Cost at parameters (loaded from ex4weights):\n(this value should be about 0.287629)\n", J)

  J, _ = nnCostFunction(θ, s, X, y, 1)
  println("Cost at parameters (loaded from ex4weights):\n(this value should be about 0.383770)\n", J)

  #==#

  println("Checking Backpropagation ...")
  checkNNGradients(0)

  println("Checking Backpropagation (w/ Regularization) ...")
  checkNNGradients(3)

  #==#

  Θ = trainNN(s, X, y, λ; iterations = 50)

  #==#

  p = predict(Θ, X)
  println("Training Set Accuracy: ", mean(p .== y) * 100)
end
