include("costFunction.jl")

function cost(X, y, λ)
  m = size(X, 1)
  (θ) -> mean(-(y).*log(h(θ, X))' - (!y).*log(1-h(θ, X))') + (λ/(2*m))*(θ[2:end]'*θ[2:end])[1]
end

function gradient!(X, y, λ)
  function (θ, dθ)
    gradient!(X, y)(θ, dθ)
    m = size(X, 1)
    copy!(dθ, dθ + (λ/m)*θ)
    dθ[1] -= (λ/m)*θ[1]
  end
end

function costFunctionReg(θ, X, y, λ)
  grad = zeros(θ)

  J = cost(X, y, λ)(θ)
  gradient!(X, y, λ)(θ, grad)
  
  J, grad
end
