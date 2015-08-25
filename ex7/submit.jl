using HTTPClient.HTTPC
using JSON
using MAT

include("ex7.jl")
include("ex7_pca.jl")

global submissionUrl = "https://www-origin.coursera.org/api/onDemandProgrammingImmediateFormSubmissions.v1"
global email = "robert.danitz@posteo.de"
global token = "oIEU9dXnTJ6IkZvy"

myprint(v) = join(map(x -> (@sprintf "%0.5f " x), v))
myprint2(v) = join(map(x -> (@sprintf "%d " x), v))

function submit()
  X = reshape(sin(1:165), 15, 11)
  Z = reshape(cos(1:121), 11, 11)
  C = Z[1:5,:]
  idx = (1 + map(i -> i % 3, 1:15))

  request = { 
    "assignmentSlug" => "k-means-clustering-and-pca",
    "itemName" => "K-Means Clustering and PCA",
    "submitterEmail" => email,
    "secret" => token,
    "parts" => {
      "1" => { "output" => myprint(findClosestCentroids(X, C)[:]) },
      "2" => { "output" => myprint(computeCentroids(X, idx, 3)) },
      "3" => { "output" => begin
        U, S, _ = pca(X)
        myprint(abs([U[:], diagm(S)[:]])) 
      end },
      "4" => { "output" => myprint(projectData(X, Z, 5)'[:]) },
      "5" => { "output" => myprint(recoverData(X[:,1:5]', Z, 5)[:]) }
    }
  }
  response = post(submissionUrl, { "jsonBody" => JSON.json(request) })
  msg = JSON.parse(bytestring(response.body))
  display(msg)
  response
end
