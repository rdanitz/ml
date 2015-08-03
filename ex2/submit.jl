using HTTPClient.HTTPC
using JSON

include("costFunctionReg.jl")
include("predict.jl")

global submissionUrl = "https://www-origin.coursera.org/api/onDemandProgrammingImmediateFormSubmissions.v1"
#=global submissionUrl = "http://localhost:3000"=#
global email = "robert.danitz@posteo.de"
global token = "g7vlHciKqgM2mQME"

function myprint(v)
  join(map(x -> @sprintf("%0.5f ", x), v))
end

function submit()
  X = [ones(20,1) exp(1) * sin(1:20) exp(.5) * cos(1:20)]
  y = sin(X[:,1] + X[:,2]) .> 0

  θ = [.25, .5, -.5]
  λ = .1
  cost1, grad1 = costFunction(θ, X, y)
  cost2, grad2 = costFunctionReg(θ, X, y, λ)

  request = { 
    "assignmentSlug" => "logistic-regression",
    "itemName" => "Logistic Regression",
    "submitterEmail" => email,
    "secret" => token,
    "parts" => {
      "1" => { "output" => myprint(sigmoid(X)) },
      "2" => { "output" => myprint(cost1) },
      "3" => { "output" => myprint(grad1) },
      "4" => { "output" => myprint(predict(θ, X)) },
      "5" => { "output" => myprint(cost2) },
      "6" => { "output" => myprint(grad2) }
    }
  }
  response = post(submissionUrl, { "jsonBody" => JSON.json(request) })
  msg = JSON.parse(bytestring(response.body))
  println(msg)
  response
end
