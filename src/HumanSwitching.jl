module HumanSwitching

using Parameters
using StaticArrays
using LinearAlgebra
using Random
using Distributions
using Statistics

# visualization
using Blink
using Cairo:
  CairoRGBSurface,
  write_to_png
using Compose

# POMDP libraries
using POMDPs
using POMDPPolicies
import POMDPModelTools: render

export
  AgentState,
  RoomRep,
  corner_states,
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
  render_scene_compose,
  render_scene_svg,
  render_scene_blink,
  render
include("visualize.jl")

export
  rand_astate
include("utils.jl")

end # module
