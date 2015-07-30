using DataFrames
using Optim

include("mapFeature.jl")
include("costFunctionReg.jl")
include("predict.jl")
include("plotDecisionBoundary.jl")

function run()
  data = readtable("ex2data2.txt", header=false, 
                   truestrings=["1"], falsestrings=["0"], eltypes=[Float64, Float64, Bool])
  X = convert(Array, data[:, [1, 2]])
  y = convert(Array, data[:, 3])
  degree = 6
  λ = 1
  iterations = 1000

  X = mapFeature(X[:,1], X[:,2], degree)
  θ0 = zeros(size(X, 2))

  c, g = costFunctionReg(θ0, X, y, λ)
  println("Cost at initial θ (zeros): ", c)
  
  res = optimize(cost(X, y, λ), gradient!(X, y, λ), θ0, 
                 method = :bfgs,
                 #=show_trace = true,=#
                 iterations = iterations)
  θ = res.minimum
  println("Cost at θ found by optimize: ", res.f_minimum)
  
  p = predict(θ, X);
  println("Train Accuracy: ", mean(map(float, p .== y)) * 100)

  plotDecisionBoundary(θ, data)
end
