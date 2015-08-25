function featureNormalize(X)
  μ = mean(X, 1)
  σ = std(X, 1)
  X_norm = (X .- μ) ./ σ

  X_norm, μ, σ
end
