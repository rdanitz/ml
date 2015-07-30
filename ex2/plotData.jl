using Gadfly

function plotData(θ, data)
  p = plot(layer(data, x="x1", y="x2", color="x3", Geom.point), 
           layer(x->-(θ[2]*x + θ[1])/θ[3], minimum(data[:,2])-2, maximum(data[:,2])+2, Geom.line, Theme(default_color=color("red"))),
           Guide.xlabel("Exam 1 score"), Guide.ylabel("Exam 2 score"), Guide.colorkey("Admission"))
  draw(PNG("ex2.png", 20cm, 15cm), p)
end
