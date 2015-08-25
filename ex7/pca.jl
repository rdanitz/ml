function pca(X)
  m = size(X, 1)
  Σ = (1/m)X'X
  svd(Σ)
end
