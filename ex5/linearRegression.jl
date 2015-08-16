h(θ, X) = X*θ

function featureNormalize(X)
  μ = mean(X, 1)
  σ = std(X, 1)
  X_norm = (X .- μ) ./ σ

  X_norm, μ, σ
end

polyFeatures(X, p) =
  reduce((akk, i)->[akk X.^i], X, 2:p)

cost(X, y) =
  (θ) -> (1/2)*mean((h(θ, X) - y).^2)

function cost(X, y, λ)
  m = size(X, 1)
  (θ) -> cost(X, y)(θ) + (λ/(2*m))*(θ[2:end]'*θ[2:end])[1]
end

gradient!(X, y) =
  (θ, dθ) -> copy!(dθ, ((h(θ, X)-y)'*X) ./ size(X, 1))

gradient!(X, y, λ) =
  function (θ, dθ)
    gradient!(X, y)(θ, dθ)
    m = size(X, 1)
    copy!(dθ, dθ + [0, ((λ/m)*θ[2:end])])
  end

function linearRegression(θ, X, y, λ)
  grad = zeros(θ)

  J = cost(X, y, λ)(θ)
  gradient!(X, y, λ)(θ, grad)
  
  J, grad
end
