using Pkg

if !haskey(Pkg.installed(), "HumanSwitching")
  # load the environment if not yet done
  jenv = joinpath(dirname(@__FILE__()), "../.")
  Pkg.activate(jenv)
  @info("Activated Environment")
end

using ParticleFilters
using POMDPs
using POMDPPolicies
using POMDPSimulators
using POMDPGifs
using BeliefUpdaters
using POMCPOW
using MCTS

using Blink
using Revise
using HumanSwitching
const HS = HumanSwitching
using Printf
using Compose
using Random
using ProgressMeter
using D3Trees

include("estimate_value_policies.jl")

function test_mdp_solver(n_runs::Int=1)
  rng = MersenneTwister(1)

  mdp_exact = generate_non_trivial_scenario(ExactPositionSensor(),
                                            HSGaussianNoisePTT(pose_cov=[0.003, 0.003, 0.003]),
                                            deepcopy(rng)) |> mdp

  # @requirements_info MCTSSolver() mdp_awgn initialstate(mdp_awgn, rng)

  # for now we run the ...
  # - the planner with some stochastic version of the true dynamics
  # - the simulator with the true dynamics unknown to the planner
  solver = MCTSSolver(estimate_value=free_space_estimate, n_iterations=5000, depth=10, exploration_constant=40.0)
  planner = solve(solver, mdp_exact)
  simulator = HistoryRecorder(rng=rng, max_steps=100)

  for i_run in 1:n_runs
    sim_hist = simulate(simulator, mdp_exact, planner)
    makegif(mdp_exact, sim_hist, filename=joinpath(@__DIR__, "../renderings/out_mcts$i_run.gif"), extra_initial=true, show_progress=true)
    println("Discounted reward: $(discounted_reward(sim_hist))")
  end
end

function demo_pomdp(runs)
  for i_run in runs
    rng = MersenneTwister(i_run)
    # setup models
    simulation_model = generate_non_trivial_scenario(ExactPositionSensor(),
                                                     HSGaussianNoisePTT(pose_cov=[0.003, 0.003, 0.003]),
                                                     deepcopy(rng))
    planning_model = generate_hspomdp(NoisyPositionSensor(),
                                      HSGaussianNoisePTT(pose_cov=[0.1, 0.1, 0.1]),
                                      simulation_model,
                                      deepcopy(rng))

    # setup POMDP solver and belief updater
    belief_updater = SIRParticleFilter(planning_model, 10000, rng=deepcopy(rng))
    solver = POMCPOWSolver(tree_queries=5000, max_depth=100, criterion=MaxUCB(40),
                           estimate_value=free_space_estimate, default_action=zero(HSAction), rng=deepcopy(rng))
    planner = solve(solver, planning_model)

    # run simulation and render gif
    try
      @info "Run #$i_run"
      simulator = HistoryRecorder(max_steps=100, show_progress=true, rng=deepcopy(rng))
      sim_hist = simulate(simulator, simulation_model, planner, belief_updater)
      makegif(simulation_model, sim_hist, filename=joinpath(@__DIR__, "../renderings/out_pomcpow_$i_run.gif"), extra_initial=true, show_progress=true)
      println(AgentPerformance(simulation_model, sim_hist))
    catch ex
      print(ex)
    end
  end
end

function test_custom_particle_filter(runs)
  for i_run in runs
    rng = MersenneTwister(i_run)
    # setup models
    simulation_model = generate_non_trivial_scenario(ExactPositionSensor(),
                                                     HSGaussianNoisePTT(pose_cov=[0.01, 0.01, 0.01],
                                                                        goal_change_prob=0.1),
                                                     deepcopy(rng))

    planning_model = generate_hspomdp(NoisyPositionSensor(),
                                      HSGaussianNoisePTT(pose_cov=[0.01, 0.01, 0.01],
                                                         goal_change_prob=0.1),
                                      simulation_model,
                                      deepcopy(rng))

    n_particles = 2000
    # the blief updater is run with a stocahstic version of the world
    # belief_updater = SIRParticleFilter(planning_model, n_particles, rng=deepcopy(rng))
    # belief_updater = SharedExternalStateFilter(planning_model, n_particles, rng=deepcopy(rng))
    belief_updater = BasicParticleFilter(planning_model, SharedExternalStateResampler(n_particles), n_particles, deepcopy(rng))
    # the policy plannes without a model as it is always the same action
    solver = POMCPOWSolver(tree_queries=6000, max_depth=70, criterion=MaxUCB(80),
                           k_action=4, alpha_action=0.1,
                           k_observation=1, alpha_observation=0,
                           estimate_value=free_space_estimate, default_action=zero(HSAction), rng=deepcopy(rng))
    planner = solve(solver, planning_model)

    # the simulator uses the exact dynamics (not known to the belief_updater)
    simulator = HistoryRecorder(max_steps=100, show_progress=true, rng=deepcopy(rng))
    sim_hist = simulate(simulator, simulation_model, planner, belief_updater)

    # a, info = action_info(planner, initialstate_distribution(model), tree_in_info=true)
    # inchrome(D3Tree(info[:tree], init_expand=3))

    println(AgentPerformance(simulation_model, sim_hist))
    makegif(simulation_model, sim_hist, filename=joinpath(@__DIR__, "../renderings/out_pomcpow_$i_run.gif"), extra_initial=true, show_progress=true)
  end
end

# isolating a problematic / critical situation to see why / how the decision is made
function tune_policy(;make_gif::Bool=false, show_tree::Bool=false)
  i_run = 1
  critical_time = 9
  debug_out_path = joinpath(@__DIR__, "../renderings/tune_policy/")

  rng = MersenneTwister(i_run)
  # setup models
  simulation_model = generate_non_trivial_scenario(ExactPositionSensor(),
                                                   HSGaussianNoisePTT(pose_cov=[0.01, 0.01, 0.01],
                                                                      goal_change_prob=0.01),
                                                   deepcopy(rng))

  planning_model = generate_hspomdp(NoisyPositionSensor(),
                                    HSGaussianNoisePTT(pose_cov=[0.01, 0.01, 0.01],
                                                       goal_change_prob=0.01),
                                    simulation_model,
                                    deepcopy(rng))

  n_particles = 2000
  # the blief updater is run with a stocahstic version of the world
  belief_updater = BasicParticleFilter(planning_model, SharedExternalStateResampler(n_particles), n_particles, deepcopy(rng))
  # the policy plannes without a model as it is always the same action
  solver = POMCPOWSolver(tree_queries=6000, max_depth=70, criterion=MaxUCB(80),
                         k_action=4, alpha_action=0.1,
                         k_observation=1, alpha_observation=0,
                         estimate_value=free_space_estimate, default_action=zero(HSAction), rng=deepcopy(rng))
  planner = solve(solver, planning_model)

  # the simulator uses the exact dynamics (not known to the belief_updater)
  simulator = HistoryRecorder(max_steps=100, show_progress=true, rng=deepcopy(rng))
  sim_hist = simulate(simulator, simulation_model, planner, belief_updater)

  println(AgentPerformance(simulation_model, sim_hist))
  if make_gif
    makegif(simulation_model, sim_hist, filename=joinpath(debug_out_path, "out$i_run.gif"), extra_initial=true, show_progress=true)
  end

  # isolating the critical step
  step = collect(eachstep(sim_hist))
  debug_step = step[critical_time]

  open(joinpath(debug_out_path, "debug_step.png"), "w") do f
    show(f, MIME("image/png"), render(simulation_model, debug_step))
  end

  if show_tree
    a, info = action_info(planner, debug_step.b, tree_in_info=true)
    inchrome(D3Tree(info[:tree], init_expand=5))
  end
  return (planner, debug_step.b)
end

function tree(p, b, init_depth=3)
  a, info = action_info(p, b, tree_in_info=true)
  inchrome(D3Tree(info[:tree], init_expand=init_depth))
end
