using Gadfly

set_default_plot_size(20cm, 15cm)

function plotData(data)
  plot(data, x="X1", y="X2", color="y", Geom.point, 
       Guide.xlabel(""), Guide.ylabel(""), Guide.colorkey("y"))
end
