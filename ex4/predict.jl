include("nnCostFunction.jl")

predict(Θ, X) =
  mapslices(indmax, reduce((x, θ)->sigmoid([ones(size(x, 1)) x] * θ'), X, Θ), 2)
