function estimateGaussian(X)
  μ = mapslices(mean, X, 1)
  σ² = mean((X.-μ).^2, 1)
  μ[:], σ²[:]
end
