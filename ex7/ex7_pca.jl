using MAT
using Gadfly

include("featureNormalize.jl")
include("pca.jl")
include("projectData.jl")
include("recoverData.jl")
include("displayData.jl")

function run()
  data = matread("ex7data1.mat")
  X = data["X"]

  #==#

  X_norm, μ, σ = featureNormalize(X)
  U, S, _ = pca(X_norm)
  
  Gadfly.plot(layer(x=X[:, 1], y=X[:, 2], Geom.point),
              layer(x = [μ' μ'+U[:,1]][1,:], y = [μ' μ'+U[:,1]][2,:], Geom.line),
              layer(x = [μ' μ'+U[:,2]][1,:], y = [μ' μ'+U[:,2]][2,:], Geom.line))

  @printf "Top eigenvector: \n"
  @printf " U[:,1] = %f %f \n" U[1,1] U[2,1]
  @printf "(you should expect to see -0.707107 -0.707107)\n\n"

  #==#

  Gadfly.plot(x=X_norm[:,1], y=X_norm[:,2])

  K = 1
  Z = projectData(X_norm, U, K)
  @printf "Projection of the first example: %f\n" Z[1]
  @printf "(this value should be about 1.481274)\n\n"

  X_rec  = recoverData(Z, U, K)
  @printf "Approximation of the first example: %f %f\n" X_rec[1,1] X_rec[1,2]
  @printf "(this value should be about  -1.047419 -1.047419)\n\n"

  layers = [layer(x=X_rec[:,1], y=X_rec[:,2], Geom.point, Theme(default_color=color("red"))),
            [let 
               line = [X_norm[i,:], X_rec[i,:]]
               layer(x=line[:,1], y=line[:,2], Geom.point, Geom.line) 
             end for i in 1:size(X_norm, 1)]]
  apply(Gadfly.plot, layers)

  #==#

  data = matread("ex7faces.mat")
  X = data["X"]
  
  figure(name = "Original Faces")
  displayData(X[1:100,:])

  #==#

  X_norm, μ, σ = featureNormalize(X)
  U, S, _ = pca(X_norm)

  figure(name = "Eigenfaces")
  displayData(U[:, 1:36]')

  #==#

  K = 100
  Z = projectData(X_norm, U, K)

  @printf "The projected data Z has a size of: "
  display(size(Z))

  #==#

  K = 100
  X_rec = recoverData(Z, U, K)

  figure(name = "Recovered Faces")
  displayData(X_rec[1:100,:])
end
