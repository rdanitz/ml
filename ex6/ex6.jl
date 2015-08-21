using DataFrames
using MAT
using LIBSVM

include("plotData.jl")
include("gaussianKernel.jl")
include("visualizeBoundaryLinear.jl")
include("visualizeBoundary.jl")
include("dataset3Params.jl")

function run()
  data = matread("ex6data1.mat")

  X = data["X"]
  y = data["y"]
  m = size(X, 1)

  df = DataFrame(X1=X[:,1], X2=X[:,2], y=y[:])
  #=plotData(df)=#

  #==#

  train = randbool(m)
  y = [i == 0 ? -1.0 : 1.0 for i in y]
  C = 1.0
  model = svmtrain(y[train], X[train,:]'; C=C, eps=1e-3, kernel_type=int32(0))
  p, = svmpredict(model, X[~train,:]')
  @printf "Accuracy: %.2f%%\n" mean((p .== y[~train]))*100

  visualizeBoundaryLinear(df, model)

  #==#

  x1 = [1 2 1]
  x2 = [0 4 -1]
  σ = 2
  sim = gaussianKernel(x1, x2, σ)
  @printf "Gaussian Kernel between x1 = [1; 2; 1], x2 = [0; 4; -1], σ = .5 :\n%f\n(this value should be about 0.324652)\n" sim
  
  #==#

  data = matread("ex6data2.mat")
  
  X = data["X"]
  y = data["y"]
  m = size(X, 1)

  df = DataFrame(X1=X[:,1], X2=X[:,2], y=y[:])
  #=plotData(df)=#

  #==#

  train = randbool(m)
  y = [i == 0 ? -1.0 : 1.0 for i in y]
  C = 10.0
  σ = .1
  model = svmtrain(y[train], X[train,:]'; C=C, eps=1e-3, kernel_type=int32(2), gamma=1/(2*σ^2))
  p, = svmpredict(model, X[~train,:]')
  @printf "Accuracy: %.2f%%\n" mean((p .== y[~train]))*100

  visualizeBoundary(df, model)

  #==#

  data = matread("ex6data3.mat")
  
  X = data["X"]
  y = data["y"]
  Xval = data["Xval"]
  yval = data["yval"]
  m = size(X, 1)

  df = DataFrame(X1=X[:,1], X2=X[:,2], y=y[:])
  #=plotData(df)=#

  #==#

  train = randbool(m) | randbool(m)
  y = [i == 0 ? -1.0 : 1.0 for i in y]
  yval = [i == 0 ? -1.0 : 1.0 for i in yval]
  C, σ = dataset3Params(X[train,:], y[train], Xval, yval)
  model = svmtrain(y[train], X[train,:]'; C=C, eps=1e-3, kernel_type=int32(2), gamma=1/(2*σ^2))
  p, _ = svmpredict(model, X[~train,:]')
  @printf "Accuracy: %.2f%%\n" mean((p .== y[~train]))*100

  visualizeBoundary(df, model)
end
