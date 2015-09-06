function normalizeRatings(Y, R)
  Ymean = mean(Y, 2)[:]
  Ynorm = mapslices(row->row-Ymean, Y, 1)
end
