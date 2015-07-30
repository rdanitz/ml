function predict(θ, X)
  m = size(X, 1)
  p = zeros(m, 1)
  for i in 1:m
    p[i] = h(θ, X[i,:])[1] >= 0.5 ? 1 : 0
  end
  p
end
