module HumanSwitching

using Blink
using Compose
using Parameters
using StaticArrays
using Random
using Distributions
using Statistics
using LinearAlgebra
using POMDPs
using POMDPPolicies

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
  observe_only
include("policies.jl")

export
  render_scene_compose,
  render_scene_svg,
  render_scene_blink

include("visualize.jl")

export
  rand_astate
include("utils.jl")

end # module
