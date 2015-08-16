using MAT
using Gadfly

include("linearRegression.jl")
include("trainLinearReg.jl")
include("learningCurve.jl")
include("plotFit.jl")
include("validationCurve.jl")

function printLearningCurves(X, y, Xval, yval; λ=0, title1="", title2="", short_title="out")
  m = size(X, 1)
  error_train, error_val = 
    learningCurve([ones(m, 1) X], y, [ones(size(Xval, 1), 1) Xval], yval, λ=λ)

  p = plot(layer(x=1:m, y=error_train, Geom.line, color = [fill("Train", m)]),
           layer(x=1:m, y=error_val, Geom.line, color = [fill("Cross validation", m)]),
           Guide.title(title1),
           Guide.xlabel("Number of training examples"),
           Guide.ylabel("Error"))
  draw(PNG(short_title * ".png", 20cm, 15cm), p)

  println(title2)
  println("# Training Examples\tTrain Error\tCross Validation Error\n")
  display([1:length(error_train) error_train error_val])
  println()
end

function run()
  data = matread("ex5data1.mat")

  X = data["X"]
  y = data["y"]

  m = size(X, 1)
  θ = [1.0, 1.0]

  #==#

  J, grad = linearRegression(θ, [ones(m, 1) X], y, 1)

  println("Cost at theta = [1 ; 1]: ", J)
  println("(this value should be about 303.993192)\n")

  println("Gradient at theta = [1 ; 1]: ", grad)
  println("(this value should be about [-15.303016; 598.250744])\n")

  #==#

  θ = trainLinearReg([ones(m, 1) X], y, λ=0)
  
  p = plot(layer(x=X, y=y, Geom.point),
           layer(x -> (θ' * [1, x])[1], 1.5*minimum(X), 1.5*maximum(X), Theme(default_color=color("red"))),
           Guide.xlabel("Change in water level (x)"), Guide.ylabel("Water flowing out of the dam (y)"))
  draw(PNG("ex5_linreg.png", 20cm, 15cm), p)

  #==#

  Xval = data["Xval"]
  yval = data["yval"]
  λ = 1

  printLearningCurves(X, y, Xval, yval; λ=λ,
                      title1="Linear Regression Learning Curve (λ=" * string(λ) * ")",
                      title2="Linear Regression (λ=" * string(λ) * ")",
                      short_title="ex5_learningcurves")

  #==#

  Xtest = data["Xtest"]
  ytest = data["ytest"]
  p = 8

  X_poly = polyFeatures(X, p)
  X_poly, μ, σ = featureNormalize(X_poly)
  X_poly = [ones(m) X_poly]

  X_poly_test = polyFeatures(Xtest, p);
  X_poly_test = X_poly_test .- μ
  X_poly_test = X_poly_test ./ σ
  X_poly_test = [ones(size(X_poly_test, 1)) X_poly_test]

  X_poly_val = polyFeatures(Xval, p)
  X_poly_val = X_poly_val .- μ
  X_poly_val = X_poly_val ./ σ
  X_poly_val = [ones(size(X_poly_val, 1)) X_poly_val]

  println("Normalized Training Example 1:\n", X_poly[1,:])
  println()

  #==#

  λ = 1
  θ = trainLinearReg(X_poly, y, λ=λ)

  p = plot(layer(x=X, y=y, Geom.point),
           plotFit(minimum(X), maximum(X), μ, σ, θ, p),
           Guide.title("Polynomial Regression Fit (λ = " * string(λ) * ")"),
           Guide.xlabel("Change in water level (x)"), Guide.ylabel("Water flowing out of the dam (Y)"))
  draw(PNG("ex5_polynomialreg.png", 20cm, 15cm), p)

  printLearningCurves(X_poly, y, X_poly_val, yval; λ=λ, 
                      title1="Polynomial Regression Learning Curve (λ = " * string(λ) * ")",
                      title2="Polynomial Regression (λ = " * string(λ) * ")",
                      short_title="ex5_learningcurves_poly")

  #==#

  Λ, error_train, error_val = validationCurve(X_poly, y, X_poly_val, yval)

  m = length(Λ)
  p = plot(layer(x=Λ, y=error_train, Geom.line, color = [fill("Train", m)]),
           layer(x=Λ, y=error_val, Geom.line, color = [fill("Cross validation", m)]),
           Guide.xlabel("λ"),
           Guide.ylabel("Error"))
  draw(PNG("ex5_validationcurves.png", 20cm, 15cm), p)

  println("λ\tTrain Error\tValidation Error")
  display([Λ error_train error_val])
end
