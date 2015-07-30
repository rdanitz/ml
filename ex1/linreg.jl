using Gadfly

set_default_plot_size(20cm, 15cm)

function warmUpExercise()
  eye(5)
end

function computeCost(X, y, θ)
  J = 0
  m = length(y)
  for i in 1:m
    J += (θ * X[i,:]' - y[i])^2;
  end
  J = J/(2*m)
end

function gradientDescent(X, y, θ, α, iterations)
  J_hist = zeros(iterations)
  m, n = size(X)
  for iter in 1:iterations
    θ_ = copy(θ)
    for j in 1:n
      s = 0
      for i in 1:m
        s += ((θ_*X[i,:]' - y[i]) * X[i, j])[1]
      end
      θ[j] = θ_[j] - α * (s/m)
    end
    J_hist[iter] = computeCost(X, y, θ)[1]
  end
  θ, J_hist
end

function ex1()
  warmUpExercise()

  data = readcsv("ex1data1.txt")
  x = data[:,1]
  y = data[:,2]
  m = length(y)
  min = 0
  max = maximum(x)
  X = hcat(ones(m), x)
  n = size(X, 2)
  θ = zeros(n)'
  α = 0.01
  iterations = 1500

  θ, _ = gradientDescent(X, y, θ, α, iterations)
  plot(layer(x=x, y=y, Geom.point), 
       layer(x->θ[1] + θ[2]*x, min, max*1.5, Geom.line, Theme(default_color=color("red"))))
end

function featureNormalize(X)
  μ = mean(X, 1)
  σ = std(X, 1)
  X_norm = (X .- μ) ./ σ

  X_norm, μ, σ
end

function normalEqn(X, y)
  ((X'*X)^-1*X'*y)'
end

function ex1multi()
  data = readcsv("ex1data2.txt")
  X = data[:,1:2]
  y = data[:,3]
  m = length(y)
  X_norm, μ, σ = featureNormalize(X)
  X = hcat(ones(m), X_norm)
  n = size(X, 2)
  θ = zeros(n)'
  α = 0.1
  iterations = 400

  θ, J_hist = gradientDescent(X, y, θ, α, iterations)
  v = [1 (1650-μ[1])/σ[1] (3-μ[2])/σ[2]]
  price = first(θ * v')
  println("price using gradient descent: ", price)
  
  data = readcsv("ex1data2.txt")
  X = data[:,1:2]
  y = data[:,3]
  X = hcat(ones(m), X)
  θ = normalEqn(X, y)
  v = [1 1650 3]
  price = first(θ * v')
  println("price using normal equation: ", price)
end
