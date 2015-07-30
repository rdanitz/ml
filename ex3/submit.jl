using HTTPClient.HTTPC
using JSON

include("lrCostFunction.jl")
include("oneVsAll.jl")
include("predictOneVsAll.jl")
include("predict.jl")

global submissionUrl = "https://www-origin.coursera.org/api/onDemandProgrammingImmediateFormSubmissions.v1"
#=global submissionUrl = "http://localhost:3000"=#
global email = "robert.danitz@posteo.de"
global token = "MjR9ss6GyWd8EQcA"

myprint(v) = join(map(x -> @sprintf("%0.5f ", x), v))

function submit()
  X = [ones(20) exp(1)*sin(1:1:20) exp(.5)*cos(1:1:20)]
  y = sin(X[:,1] .+ X[:,2]) .> 0
  Xm = [ -1 -1 ; -1 -2 ; -2 -1 ; -2 -2 ;
          1  1 ;  1  2 ;  2  1 ;  2  2 ;
         -1  1 ; -1  2 ; -2  1 ; -2  2 ;
          1 -1 ;  1 -2 ; -2 -1 ; -2 -2 ]
  ym = [ 1 1 1 1 2 2 2 2 3 3 3 3 4 4 4 4 ]'
  t1 = sin(reshape(1:2:24, 4, 3))
  t2 = cos(reshape(1:2:40, 4, 5))
  θ = [.25, .5, -.5]

  J, grad = lrCostFunction(θ, X, y, .1)

  request = { 
    "assignmentSlug" => "multi-class-classification-and-neural-networks",
    "itemName" => "Multi-class Classification and Neural Networks",
    "submitterEmail" => email,
    "secret" => token,
    "parts" => {
      "1" => { "output" => myprint([J, grad]) },
      "2" => { "output" => myprint(oneVsAll(Xm, ym, 4, .1)) },
      "3" => { "output" => myprint(predictOneVsAll(t1, Xm)) },
      "4" => { "output" => myprint(predict(t1, t2, Xm)) }
    }
  }
  response = post(submissionUrl, { "jsonBody" => JSON.json(request) })
  msg = JSON.parse(bytestring(response.body))
  display(msg)
  response
end
