using MAT
#=using Winston=#
using Optim

include("cofiCostFunction.jl")
include("normalizeRatings.jl")
include("loadMovieList.jl")

function run()
  data = matread("ex8_movies.mat")

  Y = data["Y"]
  R = data["R"]

  @printf "Average rating for movie 1 (Toy Story): %f / 5\n\n" mean(Y[1,R[1,:][:]])

  #=imagesc(Y)=#
  #=ylabel("Movies")=#
  #=xlabel("Users")=#

  #==#

  data = matread("ex8_movieParams.mat")

  X = data["X"]
  Θ = data["Theta"]

  users = 4
  movies = 5
  n = 3

  X = X[1:movies,1:n]
  Θ = Θ[1:users,1:n]
  Y = Y[1:movies,1:users]
  R = R[1:movies,1:users]
  params = [X[:], Θ[:]]

  J, _ = cofiCostFunction(params, Y, R, size(X), size(Θ))
             
  @printf "Cost at loaded parameters: %f\n(this value should be about 22.22)\n" J

  #==#

  J, _ = cofiCostFunction(params, Y, R, size(X), size(Θ); λ=1.5)
             
  @printf "Cost at loaded parameters (λ = 1.5): %f\n(this value should be about 31.34)\n" J

  #==#

  my_ratings = zeros(Int, 1682)

  my_ratings[1] = 4
  my_ratings[7] = 3
  my_ratings[12]= 5
  my_ratings[54] = 4
  my_ratings[64]= 5
  my_ratings[66]= 3
  my_ratings[69] = 5
  my_ratings[98] = 2
  my_ratings[183] = 4
  my_ratings[226] = 5
  my_ratings[355]= 5

  #==#

  data = matread("ex8_movies.mat")

  Y = data["Y"]
  R = data["R"]

  Y = [my_ratings Y]
  R = [(my_ratings .!= 0) R]

  Ynorm, Ymean = normalizeRatings(Y, R)

  users = size(Y, 2)
  movies = size(Y, 1)
  n = 10

  X = randn(movies, n)
  Θ = randn(users, n)

  init_params = [X[:], Θ[:]]

  λ = 10

  res = optimize(cost(Ynorm, R, size(X), size(Θ), λ), 
                 gradient!(Ynorm, R, size(X), size(Θ), λ), 
                 init_params, 
                 method = :l_bfgs,
                 show_trace = true,
                 iterations=100)
  params = res.minimum

  X = reshape(params[1:reduce(*, size(X))], size(X))
  Θ = reshape(params[reduce(*, size(X))+1:end], size(Θ))

  #==#

  p = X * Θ'
  my_predictions = p[:,1] + Ymean

  movieList = loadMovieList()

  my_suggestions = sortperm(my_predictions, rev=true)
  println("\nTop recommendations for you:")
  for i in my_suggestions[1:10]
    @printf "Predicting rating %.1f for movie %s\n" my_predictions[i] movieList[i]
  end

  println("\n\nOriginal ratings provided:")
  for i in 1:length(my_ratings)
    if my_ratings[i] > 0 
      @printf "Rated %d for %s\n" my_ratings[i] movieList[i]
    end
  end
end
