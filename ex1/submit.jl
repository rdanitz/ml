using HTTPClient.HTTPC
using JSON

global submissionUrl = "https://www-origin.coursera.org/api/onDemandProgrammingImmediateFormSubmissions.v1"
#=global submissionUrl = "http://localhost:3000"=#
global email = "robert.danitz@posteo.de"
global token = "hmFHVZT3dw2FWrMV"

function myprint(v)
  join(map(x -> @sprintf("%0.5f ", x), v))
end

function submit()
  X1 = hcat(ones(20), (e + e^2 * (.1:.1:2)))
  Y1 = X1[:,2] + sin(X1[:,1]) + cos(X1[:,2]);
  X2 = [X1 X1[:,2].^.5 X1[:,2].^.25];
  Y2 = Y1.^.5 + Y1;

  request = { 
    "assignmentSlug" => "linear-regression",
    "itemName" = "Linear Regression with Multiple Variables",
    "submitterEmail" => email,
    "secret" => token,
    "parts" => {
      "1" => { "output" => myprint(eye(5)) },
      "2" => { "output" => myprint(computeCost(X1, Y1, [.5 -.5])) },
      "3" => { "output" => myprint(gradientDescent(X1, Y1, [.5 -.5], .01, 10)[1]) },
      "4" => { "output" => myprint(featureNormalize(X2[:,2:4])[1]) },
      "5" => { "output" => myprint(computeCost(X2, Y2, [.1 .2 .3 .4])) },
      "6" => { "output" => myprint(gradientDescent(X2, Y2, [-.1 -.2 -.3 -.4], .01, 10)[1]) },
      "7" => { "output" => myprint(normalEqn(X2, Y2)) }
    }
  }
  response = post(submissionUrl, { "jsonBody" => JSON.json(request) })
  msg = JSON.parse(bytestring(response.body))
  println(msg)
  response
end
