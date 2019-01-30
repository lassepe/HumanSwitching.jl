module POMotionPlanning

using Parameters

using Gtk
using Cairo

export
  Room,
  render
include("environment.jl")

export
  Rectangle,
  transform_coords,
  transform_scale
include("rendering_utils.jl")

end # module
