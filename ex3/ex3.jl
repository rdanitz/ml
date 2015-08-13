using MAT

include("lrCostFunction.jl")
include("oneVsAll.jl")
include("predictOneVsAll.jl")

function run()
  data = matread("ex3data1.mat")

  X = data["X"]
  y = data["y"]
  num_labels = 10
  λ = .1

  θs = oneVsAll(X, y, num_labels, λ)
  
  p = predictOneVsAll(θs, X)
  println("Training Set Accuracy: ", mean(p .== y) * 100)
end
