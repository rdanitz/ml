include("debugInitializeWeights.jl")
include("computeNumericalGradient.jl")

function checkNNGradients(λ=0)
  s = {3, 5, 3}
  m = 5

  Θ1 = debugInitializeWeights(s[2], s[1])
  Θ2 = debugInitializeWeights(s[3], s[2])
  X  = debugInitializeWeights(m, s[1]-1)
  y  = 1+map(i->mod(i, s[3]), 1:m)

  θ = [Θ1[:], Θ2[:]]

  costFunc = (θ) -> nnCostFunction(θ, s, X, y, λ)
  J, grad = costFunc(θ)
  numgrad = computeNumericalGradient(costFunc, θ)

  display([numgrad grad])
  println("The above two columns you get should be very similar.\n(Left-Your Numerical Gradient, Right-Analytical Gradient)\n\n")

  diff = norm(numgrad-grad)/norm(numgrad+grad)

  println("If your backpropagation implementation is correct, then \n",
          "the relative difference will be small (less than 1e-9). \n",
          "\nRelative Difference: ", diff, "\n")

end
