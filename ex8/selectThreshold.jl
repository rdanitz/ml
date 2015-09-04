function computeF1(y, p, ϵ)
  tp = sum((p .<= ϵ) & (y .== 1))
  fp = sum((p .<= ϵ) & (y .== 0))
  fn = sum((p .>  ϵ) & (y .== 1))
  precision = tp/(tp+fp)
  recall = tp/(tp+fn)
  (2precision*recall)/(precision+recall)
end

function selectThreshold(y, p)
  best_ϵ = 0
  bestF1 = 0
  F1 = 0
  step = (maximum(p)-minimum(p))/1000
  for ϵ in minimum(p):step:maximum(p)
    F1 = computeF1(y, p, ϵ)
    if F1 > bestF1
      bestF1 = F1
      best_ϵ = ϵ
    end
  end
  best_ϵ, bestF1
end
