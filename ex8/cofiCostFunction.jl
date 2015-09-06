h(Θ, X) = X*Θ'

cost(y, R, sizeX, sizeΘ) =
  function (params)
    X = reshape(params[1:reduce(*, sizeX)], sizeX)
    Θ = reshape(params[reduce(*, sizeX)+1:end], sizeΘ)
    1/2 * sum((h(Θ, X) - y).^2 .* R)
  end

cost(y, R, sizeX, sizeΘ, λ) =
  function (params)
    regX = λ/2 * sum(params[1:reduce(*, sizeX)].^2)
    regΘ = λ/2 * sum(params[reduce(*, sizeX)+1:end].^2)
    cost(y, R, sizeX, sizeΘ)(params) + regX + regΘ
  end

gradientX(X, Θ, y, R) = ((h(Θ, X) - y) .* R) * Θ
gradientX(X, Θ, y, R, λ) = gradientX(X, Θ, y, R) + λ*X

gradientΘ(X, Θ, y, R) = ((h(Θ, X) - y) .* R)' * X
gradientΘ(X, Θ, y, R, λ) = gradientΘ(X, Θ, y, R) + λ*Θ

gradient(X, Θ, y, R, λ) =
  gradientX(X, Θ, y, R, λ), gradientΘ(X, Θ, y, R, λ)

gradient!(y, R, sizeX, sizeΘ, λ) =
  function (params, grad)
    X = reshape(params[1:reduce(*, sizeX)], sizeX)
    Θ = reshape(params[reduce(*, sizeX)+1:end], sizeΘ)
    gradX, gradΘ = gradient(X, Θ, y, R, λ)
    copy!(grad, [gradX[:], gradΘ[:]])
  end

gradient!(y, R, sizeX, sizeΘ) = gradient!(y, R, sizeX, sizeΘ, 0)

function cofiCostFunction(params, y, R, sizeX, sizeΘ; λ=0)
  grad = [zeros(params)]

  J = cost(y, R, sizeX, sizeΘ, λ)(params)
  gradient!(y, R, sizeX, sizeΘ, λ)(params, grad)
  
  J, grad
end
