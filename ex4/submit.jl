using HTTPClient.HTTPC
using JSON

include("ex4.jl")

global submissionUrl = "https://www-origin.coursera.org/api/onDemandProgrammingImmediateFormSubmissions.v1"
#=global submissionUrl = "http://localhost:3000"=#
global email = "robert.danitz@posteo.de"
global token = "uxO4sjXz5MdOTLFu"

myprint(v) = join(map(x -> @sprintf("%0.5f ", x), v))

function submit()
  X = reshape(3 * sin(1:1:30), 3, 10)
  Xm = reshape(sin(1:32), 16, 2) / 5
  ym = 1 + map(i->mod(i, 4), 1:16)
  t1 = sin(reshape(1:2:24, 4, 3))
  t2 = cos(reshape(1:2:40, 4, 5))
  t  = [t1[:], t2[:]]

  request = { 
    "assignmentSlug" => "neural-network-learning",
    "itemName" => "Neural Networks Learning",
    "submitterEmail" => email,
    "secret" => token,
    "parts" => {
      "1" => { "output" => myprint(nnCostFunction(t, [2, 4, 4], Xm, ym, 0)[1]) },
      "2" => { "output" => myprint(nnCostFunction(t, [2, 4, 4], Xm, ym, 1.5)[1]) },
      "3" => { "output" => myprint(sigmoidGradient(X)) },
      "4" => { "output" => begin
        J, grad = nnCostFunction(t, [2, 4, 4], Xm, ym, 0) 
        myprint([J, grad])
      end },
      "5" => { "output" => begin 
        J, grad = nnCostFunction(t, [2, 4, 4], Xm, ym, 1.5) 
        myprint([J, grad])
      end }
    }
  }
  response = post(submissionUrl, { "jsonBody" => JSON.json(request) })
  msg = JSON.parse(bytestring(response.body))
  display(msg)
  response
end
