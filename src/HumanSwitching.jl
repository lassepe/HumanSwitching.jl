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
using POMDPSimulators
using POMDPPolicies
using ParticleFilters

# packages that are extended by this module
import ParticleFilters # modified in particle_fitler.jl
import POMDPs # modified in particle_filter.jl
import POMDPModelTools: render # modified in visualization.jl

export
  Pose,
  RoomRep,
  HSState,
  HSAction,
  HSActionSpace,
  HSMDP,
  HSPOMDP,
  HSModel,
  HSSensor,
  HSRewardModel,
  HSPostTransitionTransform,
  HSIdentityPTT,
  HSGaussianNoisePTT,
  ExactPositionSensor,
  NoisyPositionSensor,
  mdp,
  room,
  hbm,
  external,
  human_pose,
  human_target,
  robot_pose,
  robot_target,
  generate_s,
  generate_o,
  initialstate,
  reward,
  reward_model,
  discount,
  apply_action
include("pomdp_formulation.jl")

export
  HSTransitionModel,
  generate_s
include("reward_model.jl")

export
  PControlledHumanTransition,
  PControlledHumanAWGNTransition
include("transition_models.jl")

export
  generate_hspomdp,
  generate_non_trivial_scenario
include("problem_gen.jl")

export
  render_step_compose,
  render_step_svg,
  render_step_blink,
  render
include("visualize.jl")

export
  rand_astate,
  dist_to_pose,
  robot_dist_to_target,
  corner_poses,
  has_collision
include("utils.jl")

export
  AgentPerformance
include("agent_performance_metrics.jl")

end # module
