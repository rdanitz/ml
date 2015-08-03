sigmoid(z) = 1./(1+exp(-z))
h(θ, X) = sigmoid(X * θ)

function cost(X, y)
  (θ) -> mean(-(y).*log(h(θ, X)) - (!y).*log(1-h(θ, X)))
end

function gradient!(X, y)
  (θ, dθ) -> copy!(dθ, ((h(θ, X) - y)' * X) ./ size(X,1))
end

function costFunction(θ, X, y)
  grad = zeros(θ)

  J = cost(X,y)(θ)
  gradient!(X,y)(θ,grad)
  
  J, grad
end
