module HumanSwitching

using Parameters
using Compose
using StaticArrays

using Random
using Distributions
using Statistics
using LinearAlgebra

using POMDPs

export
  AgentState,
  RoomRep
include("pomdp_formulation.jl")

export
  render_scene
include("visualize.jl")

include("utils.jl")

end # module
