include("sigmoidGradient.jl")

pairs(coll) = map(i -> (coll[i-1], coll[i]), 2:length(coll))

unroll(Θ) = [Θ[1][:], Θ[2][:]]
roll(θ, s) = {reshape(θ[1:s[2]*(s[1]+1)], s[2], s[1]+1),
              reshape(θ[1+s[2]*(s[1]+1):end], s[3], s[2]+1)}

function h(Θ, X)
  a1 = [ones(size(X, 1)) X]
  z2 = a1*Θ[1]'
  a2 = sigmoid(z2)
  a2 = [ones(size(a2, 1)) a2]
  z3 = a2*Θ[2]'
  a3 = sigmoid(z3)
  hΘ = a3
  hΘ, z2, a2, z3
end

function cost(s, X, y)
  function (θ)
    Θ = roll(θ, s)
    -mean((y).*log(h(Θ, X)[1]) + (1-y).*log(1-h(Θ, X)[1]))
  end
end

function cost(s, X, y, λ)
  m = size(X, 1)
  function (θ)
    Θ = roll(θ, s)
    cost(s, X, y)(θ)*s[end] + 
    (λ/(2*m))*sum(map(i->sum(diag(i[:,2:end]*i[:,2:end]')), Θ))
  end
end

function gradient!(s, X, y, λ)
  m = size(X, 1)
  function (θ, dθ)
    Θ = roll(θ, s)

    a1 = [ones(size(X, 1)) X]
    hΘ, z2, a2, _ = h(Θ, X) 
    δ3 = (hΘ - y)'
    δ2 = (Θ[2][:,2:end]' * δ3) .* sigmoidGradient(z2)'
    Δ1 = δ2 * a1
    Δ2 = δ3 * a2

    grad = {(1/m)*Δ1 + (λ/m)*[zeros(size(Θ[1], 1)) Θ[1][:,2:end]],
            (1/m)*Δ2 + (λ/m)*[zeros(size(Θ[2], 1)) Θ[2][:,2:end]]}

    copy!(dθ, unroll(grad))
  end
end

function nnCostFunction(θ, s, X, y, λ = 0)
  Θ = roll(θ, s)
  m = size(X,1)
  y = sparse(collect(1:m), convert(Array{Int64,1}, collect(y)), true)
  y = full(y)
  J = cost(s, X, y, λ)(θ)

  grad = unroll({zeros(Θ[1]), zeros(Θ[2])})
  gradient!(s, X, y, λ)(θ, grad)

  J, grad
end
