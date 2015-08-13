function debugInitializeWeights(fan_out, fan_in)
  W = zeros(fan_out, fan_in+1)
  W = reshape(sin(1:length(W)), size(W)) / 10
end
