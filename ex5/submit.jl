using HTTPClient.HTTPC
using JSON

include("ex5.jl")

global submissionUrl = "https://www-origin.coursera.org/api/onDemandProgrammingImmediateFormSubmissions.v1"
#=global submissionUrl = "http://localhost:3000"=#
global email = "robert.danitz@posteo.de"
global token = "jxpOE2nNTn220QHk"

myprint(v) = join(map(x -> @sprintf("%0.5f ", x), v))

function submit()
  X = [ones(10) sin(1:1.5:15) cos(1:1.5:15)]
  y = sin(1:3:30)
  Xval = [ones(10) sin(0:1.5:14) cos(0:1.5:14)]
  yval = sin(1:10)

  request = { 
    "assignmentSlug" => "regularized-linear-regression-and-bias-variance",
    "itemName" => "Regularized Linear Regression and Bias/Variance",
    "submitterEmail" => email,
    "secret" => token,
    "parts" => {
      "1" => { "output" => myprint(linearRegression([.1, .2, .3], X, y, .5)[1]) },
      "2" => { "output" => myprint(linearRegression([.1, .2, .3], X, y, .5)[2]) },
      "3" => { "output" => begin
        error_train, error_val = learningCurve(X, y, Xval, yval, λ=1)
        myprint([error_train[:], error_val[:]])
      end },
      "4" => { "output" => myprint(polyFeatures(X[2,:], 8)) },
      "5" => { "output" => begin 
        Λ, error_train, error_val = validationCurve(X, y, Xval, yval)
        myprint([Λ[:], error_train[:], error_val[:]])
      end }
    }
  }
  response = post(submissionUrl, { "jsonBody" => JSON.json(request) })
  msg = JSON.parse(bytestring(response.body))
  display(msg)
  response
end

