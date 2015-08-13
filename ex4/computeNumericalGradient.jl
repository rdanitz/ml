function computeNumericalGradient(J, θ)
  numgrad = zeros(size(θ))
  perturb = zeros(size(θ))
  e = 1e-4

  for p in 1:length(θ)
      perturb[p] = e
      loss1, _ = J(θ - perturb)
      loss2, _ = J(θ + perturb)
      numgrad[p] = (loss2-loss1) / (2*e)
      perturb[p] = 0
  end
  numgrad
end
