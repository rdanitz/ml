function emailFeatures(indices)
  df = readtable("vocab.txt", separator='\t', header=false)
  n = length(df[:x2])
  x = zeros(Int, n)

  for i in indices
    x[i] = 1 
  end

  x
end
