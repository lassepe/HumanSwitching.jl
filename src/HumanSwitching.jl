module HumanSwitching

using Parameters
using Compose
using StaticArrays

using POMDPs

export
  AgentState,
  RoomRep
include("environment.jl")

export
  render_scene
include("visualize.jl")

end # module
