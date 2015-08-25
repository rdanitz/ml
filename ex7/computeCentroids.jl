function computeCentroids(X, idx, K)
  centroid(X, idx, k) = 
    #=mean(reduce((akk, i) -> idx[i] == k ? [akk, {X[i,:]}] : akk, {}, 1:length(idx)))=#
    sum(X .* (idx .== k), 1)/countnz(idx .== k)
  centroids = map(k -> centroid(X, idx, k), 1:K)
  apply(vcat, centroids)
end
