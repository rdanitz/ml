include("costFunction.jl")

function cost(X, y, λ)
  m = size(X, 1)
  (θ) -> cost(X, y)(θ) + (λ/(2*m))*(θ[2:end]'*θ[2:end])[1]
end

function gradient!(X, y, λ)
  m = size(X, 1)
  function (θ, dθ)
    gradient!(X, y)(θ, dθ)
    copy!(dθ, dθ + [0, ((λ/m)*θ)[2:end]])
  end
end

function lrCostFunction(θ, X, y, λ)
  grad = zeros(θ)

  J = cost(X, y, λ)(θ)
  gradient!(X, y, λ)(θ, grad)
  
  J, grad
end
