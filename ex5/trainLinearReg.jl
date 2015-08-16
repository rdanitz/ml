using Optim

include("linearRegression.jl")

function trainLinearReg(X, y; λ = 0, iterations = 400)
  initθ = zeros(size(X, 2))

  res = optimize(cost(X, y, λ), gradient!(X, y, λ), initθ,
                 method = :bfgs,
                 #=show_trace = true,=#
                 iterations = iterations)
  res.minimum
end
