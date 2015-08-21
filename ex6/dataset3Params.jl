function dataset3Params(X, y, Xval, yval)
  C = 1.0
  σ = .3
  maxp = 0

  for i in [.01, .03, .1, .3, 1, 3, 10, 30], 
      j in [.01, .03, .1, .3, 1, 3, 10, 30]
    model = svmtrain(y, X'; C=i, eps=1e-3, kernel_type=int32(2), gamma=1/(2*j^2))
    p, _ = svmpredict(model, Xval')
    if sum(p .== yval) > maxp
      C = i
      σ = j
      maxp = sum(p .== yval)
    end
  end

  C, σ
end
