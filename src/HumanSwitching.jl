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
  RoomRep,
  HSState,
  HSAction,
  HSMDP,
  HSPOMDP,
  HSModel,
  room,
  generate_s,
  generate_o,
  initialstate,
  reward,
  discount
include("pomdp_formulation.jl")

export
  render_scene
include("visualize.jl")

export
  rand_astate
include("utils.jl")

end # module
