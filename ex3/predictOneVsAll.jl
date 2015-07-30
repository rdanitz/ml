include("costFunction.jl")

predictOneVsAll(θs, X) =
  mapslices(indmax, sigmoid([ones(size(X,1)) X] * θs'), 2)
