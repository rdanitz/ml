using Winston

function displayData(X, height=20, width=20)
  colormap("grays")
  figure()
  h = imagesc(reshape(X, height, width))
  display(h)
end
