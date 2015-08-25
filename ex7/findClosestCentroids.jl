findClosestCentroid(X, centroids) =
  indmin(mapslices(norm, repmat(X, size(centroids, 1), 1) .- centroids, 2).^2)

findClosestCentroids(X, centroids) =
  mapslices(x -> findClosestCentroid(x', centroids), X, 2)
