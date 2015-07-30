using Optim

include("lrCostFunction.jl")

function oneVsAll(X, y, num_labels, λ, iterations = 400)
  m, n = size(X)
  θs = zeros(num_labels, n+1)
  X = hcat(ones(m, 1), X)

  for i in 1:num_labels
    res = optimize(cost(X, (y .== i), λ), gradient!(X, (y .== i), λ), collect(θs[i,:]),
                   method = :bfgs,
                   #=show_trace = true,=#
                   iterations = iterations)
    θs[i,:] = res.minimum
  end
  θs
end
