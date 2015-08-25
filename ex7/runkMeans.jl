include("plotProgressMeans.jl")

function step(X, akk, K; history = false)
  if history
    idx = findClosestCentroids(X, first(first(akk)))
    centroids = computeCentroids(X, idx, K)
    ([{centroids}, first(akk)], idx)
  else
    idx = findClosestCentroids(X, first(akk))
    centroids = computeCentroids(X, idx, K)
    (centroids, idx)
  end
end

function runkMeans(X, initial_centroids, max_iters; plot_progress = false)
  K = size(initial_centroids, 1)
  if plot_progress
    centroids_history, idx = reduce((akk, _) -> step(X, akk, K, history = true), ({initial_centroids}, None), 1:max_iters)
    plotProgressMeans(X, centroids_history[2:end], idx, K)
    first(centroids_history), idx
  else
    centroids, idx = reduce((akk, _) -> step(X, akk, K, history = false), (initial_centroids, None), 1:max_iters)
    centroids, idx
  end
end
