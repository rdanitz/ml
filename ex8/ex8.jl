using MAT
using Gadfly

include("estimateGaussian.jl")
include("multivariateGaussian.jl")
include("visualizeFit.jl")
include("selectThreshold.jl")

function run()
  data = matread("ex8data1.mat")
  X = data["X"]
  Xval = data["Xval"]
  yval = data["yval"]

  plot(x=X[:, 1], y=X[:, 2], Geom.point,
       Guide.xlabel("Latency (ms)"),
       Guide.ylabel("Throughput (mb/s)"))

  #==#

  μ, σ² = estimateGaussian(X)
  p = multivariateGaussian(X, μ, σ²)
  visualizeFit(X, μ, cov(X))

  #==#

  pval = multivariateGaussian(Xval, μ, σ²)
  ϵ, F1 = selectThreshold(yval, pval)

  @printf "Best epsilon found using cross-validation: %e\n" ϵ
  @printf "Best F1 on Cross Validation Set: %f\n" F1
  @printf "   (you should see a value epsilon of about 8.99e-05)\n\n"

  outliers = p .<= ϵ
  plot(layer(x=X[outliers,1], y=X[outliers,2], Geom.point, Theme(default_color=colorant"red")),
       layer(x=X[!outliers,1], y=X[!outliers,2], Geom.point))

  #==#

  data = matread("ex8data2.mat")
  X = data["X"]
  Xval = data["Xval"]
  yval = data["yval"]

  μ, σ² = estimateGaussian(X)
  p = multivariateGaussian(X, μ, σ²)
  pval = multivariateGaussian(Xval, μ, σ²)

  ϵ, F1 = selectThreshold(yval, pval)

  @printf "Best epsilon found using cross-validation: %e\n" ϵ
  @printf "Best F1 on Cross Validation Set:  %f\n" F1
  @printf "# Outliers found: %d\n" sum(p .< ϵ)
  @printf "   (you should see a value epsilon of about 1.38e-18)\n\n"
end
