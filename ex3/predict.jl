include("costFunction.jl")

predict(Θ1, Θ2, X) =
  mapslices(indmax, reduce((x,θ)->sigmoid([ones(size(X, 1)) x] * θ'), X, {Θ1 Θ2}), 2)
