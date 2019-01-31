module HumanSwitching

using Parameters

using Gtk
using Cairo

export
  Rectangle,
  transform_coords,
  transform_scale
include("rendering_utils.jl")

export
  Room,
  render
include("environment.jl")

end # module
