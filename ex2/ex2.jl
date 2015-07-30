using DataFrames
using Optim

include("costFunction.jl")
include("predict.jl")
include("plotData.jl")

function run()
  data = readtable("ex2data1.txt", header=false, 
                   truestrings=["1"], falsestrings=["0"], eltypes=[Float64, Float64, Bool])
  X = convert(Array, data[:, [1, 2]])
  y = convert(Array, data[:, 3])
  m, n = size(X)
  X = hcat(ones(m, 1), X)
  θ0 = zeros(n+1)

  c, g = costFunction(θ0, X, y)
  println("Cost at initial θ (zeros): ", c)
  println("Gradient at initial θ (zeros):")
  println(g)

  res = optimize(cost(X, y), gradient!(X, y), θ0, method = :l_bfgs)
  θ = res.minimum
  println("Cost at θ found by optimize: ", res.f_minimum)
  println("θ: ", θ)

  prob = sigmoid(θ' * [1 45 85]')
  println("For a student with scores 45 and 85, we predict an admission probability of ", prob[1])
  p = predict(θ, X);
  println("Train Accuracy: ", mean(map(float, p .== y)) * 100)

  plotData(θ, data)
end
