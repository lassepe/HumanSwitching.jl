module HumanSwitching

using Parameters
using StaticArrays
using LinearAlgebra
using Random
using Distributions
using Statistics
using InteractiveUtils

# visualization
using Blink
using Cairo:
  CairoRGBSurface,
  write_to_png
using Compose
using ColorSchemes, Colors

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
  # util types
  Pose,
  RoomRep,

  # post transition transformations for particle filter
  HSPostTransitionTransform,
  HSIdentityPTT,
  HSGaussianNoisePTT,

  # human transition models
  HumanBehaviorGenerator,
  HumanBehaviorModel,
  HumanConstantVelocityBehavior,
  HumanPIDBehavior,

  # Sensor / Observation Models
  HSSensor,
  ExactPositionSensor,
  NoisyPositionSensor,

  # Core POMDP types for problem formulation
  HSRewardModel,
  HSExternalState,
  HSState,
  HSAction,
  HSActionSpace,
  HSMDP,
  HSPOMDP,
  HSModel,

  # problem utilites
  mdp,
  room,
  hbm,
  external,
  human_pose,
  human_target,
  robot_pose,
  robot_target,

  # POMDP interface implementation
  generate_s,
  generate_o,
  initialstate,
  reward,
  reward_model,
  discount,
  apply_action

include("pomdp_main.jl")
include("post_transition_transform.jl")
include("reward_model.jl")
include("human_behavior_generators.jl")
include("human_transition_models.jl")

export
  rand_astate,
  dist_to_pose,
  robot_dist_to_target,
  corner_poses,
  has_collision
include("utils.jl")

export
  render_step_compose,
  render_step_svg,
  render_step_blink,
  render
include("visualize.jl")

export
  AgentPerformance
include("agent_performance_metrics.jl")

export
  SharedExternalStateBelief,
  SharedExternalStateFilter,
  SharedExternalStateResampler
include("particle_filter.jl")

export
  generate_hspomdp,
  generate_non_trivial_scenario
include("problem_gen.jl")

end # module
