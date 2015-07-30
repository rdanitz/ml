function mapFeature(X1, X2, degree)
  out = ones(size(X1[:,1]))
  for i in 1:degree
    for j in 0:i
      out = hcat(out, (X1.^(i-j)).*(X2.^j))
    end
  end
  out
end
