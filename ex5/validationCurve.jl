function validationCurve(X, y, Xval, yval)
  Λ = [0 .001 .003 .01 .03 .1 .3 1 3 10]'

  m = length(Λ)
  error_train = zeros(m)
  error_val = zeros(m)

  for i in 1:m
    θ = trainLinearReg(X, y, λ=Λ[i])
    error_train[i] = cost(X, y)(θ)
    error_val[i] = cost(Xval, yval)(θ)
  end

  Λ, error_train, error_val
end

