using MAT
using LIBSVM

include("processEmail.jl")
include("emailFeatures.jl")

function run()
  email = readall("emailSample1.txt")
  indices = processEmail(email)
  features = emailFeatures(indices)
  
  #==#

  data = matread("spamTrain.mat")
  
  X = data["X"]
  y = data["y"]
  
  y = [i == 0 ? -1.0 : 1.0 for i in y]

  C = .1
  model = svmtrain(y, X'; C=C, eps=1e-3, kernel_type=int32(0))
  p, _ = svmpredict(model, X')
  @printf "Accuracy: %.2f%%\n" mean((p .== y))*100
 
  #==#
 
  data = matread("spamTest.mat")
  Xtest = data["Xtest"]
  ytest = data["ytest"]

  ytest = [i == 0 ? -1.0 : 1.0 for i in ytest]
  p, _ = svmpredict(model, Xtest')
  @printf "Test Accuracy: %.2f%%\n" mean((p .== ytest))*100

end
