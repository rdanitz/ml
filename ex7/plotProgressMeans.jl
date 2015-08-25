using Gadfly

second = i -> i[2]

function plotProgressMeans(X, centroids_history, idx, K)
  X1 = X.*(idx.==1) 
  X2 = X.*(idx.==2) 
  X3 = X.*(idx.==3) 
  C1 = map(i -> i[1,:], centroids_history)
  C2 = map(i -> i[2,:], centroids_history)
  C3 = map(i -> i[3,:], centroids_history)
  p = plot(layer(x=X1[:,1], y=X1[:,2], Geom.point, Theme(default_color=color("red"))),
           layer(x=X2[:,1], y=X2[:,2], Geom.point, Theme(default_color=color("green"))),
           layer(x=X3[:,1], y=X3[:,2], Geom.point, Theme(default_color=color("blue"))),
           layer(x=map(first, C1), y=map(second, C1), Geom.point, Geom.line),
           layer(x=map(first, C2), y=map(second, C2), Geom.point, Geom.line),
           layer(x=map(first, C3), y=map(second, C3), Geom.point, Geom.line))
  draw(PNG("ex7.png", 20cm, 15cm), p)
end
