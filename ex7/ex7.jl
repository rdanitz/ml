using MAT
import Images, ImageView

include("findClosestCentroids.jl")
include("computeCentroids.jl")
include("runkMeans.jl")
include("kMeansInitCentroids.jl")

function run()
  data = matread("ex7data2.mat")
  X = data["X"]

  K = 3
  initial_centroids = [3. 3.; 6. 2.; 8. 5.]
  idx = findClosestCentroids(X, initial_centroids)

  @printf "Closest centroids for the first 3 examples:\n"
  display(idx[1:3])
  @printf "\n(the closest centroids should be 1, 3, 2 respectively)\n"

  #==#

  centroids = computeCentroids(X, idx, K)

  @printf "Centroids computed after initial finding of closest centroids: \n"
  display(centroids)
  @printf "\n(the centroids should be\n"
  display([ 2.428301 3.157924
            5.813503 2.633656
            7.119387 3.616684 ])

  #==#

  data = matread("ex7data2.mat")
  X = data["X"]

  K = 3
  max_iters = 10
  initial_centroids = [3. 3.; 6. 2.; 8. 5.]

  centroids, idx = runkMeans(X, initial_centroids, max_iters, plot_progress = true)

  #==#

  img = Images.imread("bird_small.png")
  A = img.data/255

  X = apply(hcat, map(i->[i.r, i.g, i.b], A[:]))'
  K = 16
  max_iters = 10
  initial_centroids = kMeansInitCentroids(X, K)

  centroids, idx = runkMeans(X, initial_centroids, max_iters)

  X_recovered = apply(vcat, map(i->centroids[i,:], idx))
  img2 = 255reshape(mapslices(i->Images.RGB(i[1], i[2], i[3]), X_recovered, 2), size(A))'
  ImageView.view(img, pixelspacing=[1,1])
  ImageView.view(img2, pixelspacing=[1,1])
end
