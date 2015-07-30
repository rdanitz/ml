using MAT

include("predict.jl")

function run()
  data = matread("ex3data1.mat")
  weights = matread("ex3weights.mat")

  X = data["X"]
  y = data["y"]
  Θ1 = weights["Theta1"]
  Θ2 = weights["Theta2"]

  predict(Θ1, Θ2, X)
end
