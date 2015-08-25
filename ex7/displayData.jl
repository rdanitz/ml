using Winston

function displayData(X; width = round(sqrt(size(X, 2))))

  Winston.colormap("Grays")

  m, n = size(X)
  height = n/width

  rows = floor(sqrt(m))
  cols = ceil(m/rows)
  pad = 0

  display_array = - ones(int(pad + rows * (height + pad)),
                         int(pad + cols * (width + pad)))

  curr_ex = 1
  for j in 1:rows
    for i in 1:cols
      if curr_ex > m
        break
      end
      max_val = maximum(abs(X[curr_ex,:]))
      display_array[int(pad + (j - 1) * (height + pad) + (1:height)),
                    int(pad + (i - 1) * (width + pad) + (1:width))] = 
              reshape(X[curr_ex,:], int(height), int(width))/max_val
      curr_ex = curr_ex + 1
    end
    if curr_ex > m
      break
    end
  end

  h = Winston.imagesc(-(display_array+1)/2*256)
  display(h)
end
