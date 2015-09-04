using Gadfly

include("multivariateGaussian.jl")

meshgrid(range) = let X = apply(vcat, map(_->collect(range)', 1:(length(range))))
  X, X'
end

function visualizeFit(X, μ, σ²)
  z = (x, y) -> (multivariateGaussian([x y], μ, σ²))[1]

  plot(layer(x=X[:,1], y=X[:,2], Geom.point),
       layer(x=linspace(0, 30, 100), y=linspace(0, 30, 100), z=z, Geom.contour()))
  #=draw(PNG("ex8.png", 20cm, 15cm), p)=#
end
