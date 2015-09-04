function multivariateGaussian(X, μ, Σ²::Array{Float64,2})
  k = length(μ)
  I = inv(Σ²)
  p = (2π)^(-k/2) * det(Σ²)^(-.5) * exp(-.5 * (diag((X.-μ') * I * (X.-μ' )')))
end

function multivariateGaussian(X, μ, σ²)
  k = length(μ)
  Σ² = diagm(σ²[:])
  I = inv(Σ²)
  p = (2π)^(-k/2) * det(Σ²)^(-.5) * exp(-.5 * (diag((X.-μ') * I * (X.-μ' )')))
end
