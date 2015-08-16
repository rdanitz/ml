using Gadfly

function plotFit(min, max, μ, σ, θ, p)
  x = min*1.2:.05:max*1.2

  X_poly = polyFeatures(x, p)
  X_poly = X_poly .- μ
  X_poly = X_poly ./ σ

  X_poly = [ones(size(x, 1)) X_poly]

  layer(x=x, y=X_poly*θ, Geom.line)
end
