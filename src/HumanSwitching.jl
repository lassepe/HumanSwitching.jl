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
using DataFrames
using DataFramesMeta
using CSV
using Gadfly:
    Gadfly,
    Geom,
    Guide,
    Coord,
    plot

# POMDP libraries
using POMDPs
using POMDPModelTools
using POMDPSimulators
using POMDPPolicies
using ParticleFilters
using ARDESPOT
using POMCPOW

# used for graph search
using DataStructures
using NearestNeighbors
using Reel
import Base: ==

# used for simulation utils
using CPUTime
using Distributed
using POMDPGifs
using D3Trees

# packages that are extended by this module
import ParticleFilters # modified in particle_fitler.jl
import POMDPs # modified in particle_filter.jl and cpu_timing_wrappers.jl
import POMDPModelTools: render, action_info, update_info # modified in visualization.jl
import MCTS: MCTS, node_tag


export
    # util types
    Pos,
    Room,

    # post transition transformations for particle filter
    HSPhysicalTransitionNoiseModel,
    HSIdentityPTNM,
    HSGaussianNoisePTNM,

    # human transition models
    HumanBehaviorState,
    HumanBehaviorModel,

    # Sensor / Observation Models
    HSSensor,
    ExactPositionSensor,
    NoisyPositionSensor,

    # Core POMDP types for problem formulation
    HSRewardModel,
    HSExternalState,
    HSHumanState,
    HSState,
    HSAction,
    HSActionSpace,
    HSMDP,
    gen_hsmdp,
    HSPOMDP,
    HSModel,

    # problem utilites
    mdp,
    room,
    hbs,
    human_behavior_model,
    external,
    human_pos,
    human_goal,
    robot_pos,
    robot_goal,

    # POMDP interface implementation
    generate_s,
    generate_o,
    initialstate,
    reward,
    reward_model,
    discount,
    apply_robot_action
include("pomdp_main.jl")

export
    # const vel
    HumanConstVelBState,
    HumanConstVelBehavior
include("human_behaviors/human_const_vel.jl")

export
    free_evolution,
    bstate_type
include("human_behaviors/human_behaviors.jl")

export
    # pid
    HumanPIDBState,
    HumanPIDBehavior
include("human_behaviors/human_pid.jl")

export
    # boltzmann
    HumanBoltzmannBState,
    HumanRewardModel,
    HumanSingleGoalRewardModel,
    gen_human_aspace,
    # boltzmann / multi goal
    HumanBoltzmannToGoalBState,
    HumanMultiGoalBoltzmann
include("human_behaviors/human_boltzmann.jl")

export
    # model mix
    HumanUniformModelMix
include("human_behaviors/human_uniform_modelmix.jl")

include("physical_transition_noise_model.jl")
include("reward_model.jl")

export
    rand_astate,
    dist_to_pos,
    robot_dist_to_goal,
    corner_positions,
    goal_index,
    has_collision,
    issuccess,
    isfailure,
    at_robot_goal,
    isinroom
include("utils.jl")

export
    Circle,
    InfiniteCone,
    ConicalFrustum,
    LineSegment,
    contains
include("geometry.jl")

export
    render_step_compose,
    render_step_svg,
    render_step_blink,
    render,
    render_plan
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
    generate_from_template,
    generate_non_trivial_scenario
include("problem_gen.jl")

export
    StraightToGoal,
    free_space_estimate
include("estimate_value_policies.jl")

export
    plot_points,
    plot_full,
    plot_problem_instance,
    extract_value_compute,
    load_data,
    check_data,
    success_rate,
    filter_by_planner,
    tail_expectation
include("analyze_results.jl")

export
    BeliefPropagator,
    ParticleBeliefPropagator,
    predict!,
    predict
include("prob_obstacle_solver/belief_propagator.jl")

export
    SearchProblem,
    start_state,
    is_goal_state,
    successors,
    SearchNode,
    InfeasibleSearchProblemError,
    cost,
    leaf_state,
    action_sequence,
    expand,
    generic_graph_search,
    astar_search,
    weighted_astar_search
include("prob_obstacle_solver/search.jl")

export
    TimedPolicy,
    TimedUpdater
include("cpu_timing_wrappers.jl")

include("switching_policies.jl")

export
    ProbObstacleSolver,
    ProbObstaclePolicy,
    visualize_plan
include("prob_obstacle_solver/prob_obstacle_solver.jl")

export
    validation_hash,
    final_state_type,
    reproduce_scenario,
    parallel_sim,
    visualize,
    tree,
    debug,
    debug_with_plan
include("simulation_utils.jl")

end # module
