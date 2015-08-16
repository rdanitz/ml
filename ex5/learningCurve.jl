include("linearRegression.jl")

function learningCurve(X, y, Xval, yval; λ = 0)
  m = size(X, 1)
  error_train = zeros(m)
  error_val = zeros(m)

  for i in 1:m
    θ = trainLinearReg(X[1:i,:], y[1:i], λ=λ)
    error_train[i] = cost(X[1:i,:], y[1:i])(θ)
    error_val[i] = cost(Xval, yval)(θ)
  end

  error_train, error_val
end
