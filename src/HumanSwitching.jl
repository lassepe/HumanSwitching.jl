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
using ParticleFilters

# packages that are extended by this module
import POMDPs # modified in filtering.jl
import POMDPModelTools: render # modified in visualization.jl

export
  Pose,
  RoomRep,
  corner_states,
  HSState,
  HSAction,
  HSActionSpace,
  HSMDP,
  HSPOMDP,
  HSModel,
  HSSensor,
  ExactPositionSensor,
  NoisyPositionSensor,
  HSTransitionModel,
  PControlledHumanTransition,
  PControlledHumanAWGNTransition,
  room,
  generate_s,
  generate_o,
  initialstate,
  reward,
  discount,
  apply_action,
  generate_hspomdp
include("pomdp_formulation.jl")

export
  render_step_compose,
  render_step_svg,
  render_step_blink,
  render
include("visualize.jl")

export
  rand_astate,
  dist_to_pose,
  robot_dist_to_target
include("utils.jl")

end # module
