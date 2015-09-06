using HTTPClient.HTTPC
using JSON
using MAT

include("ex8.jl")
include("ex8_cofi.jl")

global submissionUrl = "https://www-origin.coursera.org/api/onDemandProgrammingImmediateFormSubmissions.v1"
global email = "robert.danitz@posteo.de"
global token = "MZDbnQVyh3MiJ4of"

myprint(v) = join(map(x -> (@sprintf "%0.5f " x), v))
myprint2(v) = join(map(x -> (@sprintf "%d " x), v))

function submit()
  users = 3
  movies = 4
  n = 5
  X = reshape(sin(1:movies*n), movies, n)
  Θ = reshape(cos(1:users*n), users, n)
  Y = reshape(sin(1:2:2*movies*users), movies, users)
  R = Y .> 0.5
  pval = [abs(Y[:]), .001, 1]
  yval = [R[:], 1, 0]
  params = [X[:], Θ[:]]

  request = { 
    "assignmentSlug" => "anomaly-detection-and-recommender-systems",
    "itemName" => "Anomaly Detection and Recommender Systems",
    "submitterEmail" => email,
    "secret" => token,
    "parts" => {
      "1" => { "output" => begin
        μ, σ² = estimateGaussian(X)
        myprint([μ, σ²])
      end },
      "2" => { "output" => begin
        bestEpsilon, bestF1 = selectThreshold(yval, pval)
        myprint([bestEpsilon, bestF1])
      end },
      "3" => { "output" => begin
        J, _ = cofiCostFunction(params, Y, R, (movies, n), (users, n))
        myprint(J)
      end },
      "4" => { "output" => begin
        _, grad = cofiCostFunction(params, Y, R, (movies, n), (users, n))
        myprint(grad)
      end },
      "5" => { "output" => begin
        J, _ = cofiCostFunction(params, Y, R, (movies, n), (users, n); λ=1.5)
        myprint(J)
      end },
      "6" => { "output" => begin
        _, grad = cofiCostFunction(params, Y, R, (movies, n), (users, n); λ=1.5)
        myprint(grad)
      end }
    }
  }
  response = post(submissionUrl, { "jsonBody" => JSON.json(request) })
  msg = JSON.parse(bytestring(response.body))
  display(msg)
  response
end
