gaussianKernel(x1, x2, σ) =
  exp(-norm(x1 .- x2)^2/(2*σ^2))
