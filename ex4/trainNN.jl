using Optim

include("nnCostFunction.jl")
include("randInitializeWeights.jl")

function trainNN(s, X, y, λ; iterations = 20)
  Θ1 = randInitializeWeights(s[1], s[2])
  Θ2 = randInitializeWeights(s[2], s[3])
  initθ = [Θ1[:], Θ2[:]]

  m = size(X,1)
  y = sparse(collect(1:m), convert(Array{Int64,1}, collect(y)), true)
  y = full(y)

  res = optimize(cost(s, X, y, λ), gradient!(s, X, y, λ), initθ,
                 method = :bfgs,
                 show_trace = true,
                 iterations = iterations)
  roll(res.minimum, s)
end
