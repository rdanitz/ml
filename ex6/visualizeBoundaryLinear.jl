using Gadfly
using LIBSVM

function visualizeBoundaryLinear(data, model)
  u = linspace(minimum(data[:X1]), maximum(data[:X1]), 100)
  v = linspace(minimum(data[:X2]), maximum(data[:X2]), 100)
  z = zeros(length(u), length(v))

  for i in 1:length(u)
    for j in 1:length(v)
        z[i,j] = svmpredict(model, [u[i] v[j]]')[1][1]
    end
  end
  
  p = plot(data, x="X1", y="X2", color="y", Geom.point, 
           layer(z=z, x=u, y=v, Geom.contour(levels=[0])),
           Guide.xlabel(""), Guide.ylabel(""), Guide.colorkey("y"))
  draw(PNG("ex6_linear.png", 20cm, 15cm), p)
end
