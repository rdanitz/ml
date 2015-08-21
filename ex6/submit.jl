using HTTPClient.HTTPC
using JSON
using MAT

include("ex6.jl")
include("ex6_spam.jl")

global submissionUrl = "https://www-origin.coursera.org/api/onDemandProgrammingImmediateFormSubmissions.v1"
global email = "robert.danitz@posteo.de"
global token = "UqNsWoIov1lQGcYI"

myprint(v) = join(map(x -> (@sprintf "%0.5f " x), v))
myprint2(v) = join(map(x -> (@sprintf "%d " x), v))

function submit()
  x1 = sin(1:10)'
  x2 = cos(1:10)'
  ec = "the quick brown fox jumped over the lazy dog"
  wi = 1 + abs(round(x1 * 1863))
  wi = [wi ; wi]

  request = { 
    "assignmentSlug" => "support-vector-machines",
    "itemName" => "Support Vector Machines",
    "submitterEmail" => email,
    "secret" => token,
    "parts" => {
      "1" => { "output" => myprint(gaussianKernel(x1, x2, 2)) },
      "2" => { "output" => begin
        data = matread("ex6data3.mat")
        X = data["X"]
        y = data["y"]
        Xval = data["Xval"]
        yval = data["yval"]
        myprint(dataset3Params(X, y[:], Xval, yval[:]))
      end },
      "3" => { "output" => myprint2(processEmail(ec)) },
      "4" => { "output" => myprint2(emailFeatures(wi)) }
    }
  }
  response = post(submissionUrl, { "jsonBody" => JSON.json(request) })
  msg = JSON.parse(bytestring(response.body))
  display(msg)
  response
end
