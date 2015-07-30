using Gadfly

function plotDecisionBoundary(θ, data, degree=6)
  u = linspace(-1, 1.5, 50)
  v = linspace(-1, 1.5, 50)
  z = zeros(length(u), length(v))

  for i in 1:length(u)
    for j in 1:length(v)
        z[i,j] = (mapFeature([u[i]], [v[j]], degree)*θ)[1]
    end
  end
  
  p = plot(layer(data, x="x1", y="x2", color="x3", Geom.point),
           layer(z=z, x=u, y=v, Geom.contour(levels=[0])),
           Guide.xlabel("Microchip Test 1"), Guide.ylabel("Microchip Test 2"), Guide.colorkey("Test score"))
  draw(PNG("ex2_reg.png", 20cm, 15cm), p)
end
