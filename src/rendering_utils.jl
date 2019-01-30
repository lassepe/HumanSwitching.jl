@with_kw struct Rectangle
  top_left::Array{Float64, 1} = [0.0, 0.0]
  width::Float64 = 1.0
  height::Float64 = 1.0
end

# Transform coordinates in world frame to coordinates used for rendering
function transform_coords(pos::AbstractVector{Float64})
    x, y = pos

    # Specify dimensions of window
    h = 600
    w = 600

    # Perform conversion
    x_trans = transform_scale(x + 30.0)
    y_trans = -transform_scale(y - 20.0)
    x_trans, y_trans
end

transform_scale(l::Float64) = l * 12.0
