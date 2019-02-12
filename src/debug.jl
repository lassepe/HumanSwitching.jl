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
using MCTS

using Blink
using Revise
using HumanSwitching
const HS = HumanSwitching
using Printf
using Compose
using Random
using ProgressMeter


function get_test_problem()
  # create some test problem
  pomdp_exact = HSPOMDP(sensor=ExactPositionSensor(), mdp=HSMDP(transition_model=PControlledHumanTransition()))
  pomdp_noisy = HSPOMDP(sensor=NoisyPositionSensor([0.3, 0.3, 0.3]), mdp=HSMDP(transition_model=PControlledHumanAWGNTransition()))
  rng = MersenneTwister(1)

  return pomdp_exact, pomdp_noisy, rng
end

function simulate_with_policy(n_runs=1)
  belief_updater = NothingUpdater()
  exact_pomdp, _, rng = get_test_problem()
  policy = FunctionPolicy(x->HSAction())
  # now step over the history and render to blink using the usual method
  win = Blink.Window()

  @showprogress for i in 1:n_runs
    history = simulate(HistoryRecorder(max_steps=100), exact_pomdp, policy, belief_updater)
    for step in eachstep(history)
      render_step_blink(exact_pomdp, step, win)
      sleep(0.2)
    end
  end

  close(win)
end

function test_belief_updater(n_runs::Int=1)
  exact_pomdp, noisy_pomdp, rng = get_test_problem()
  # the blief updater is run with a stocahstic version of the world
  belief_updater = SIRParticleFilter(noisy_pomdp, 1000, rng=rng)
  # the policy plannes without a model as it is always the same action
  policy = FunctionPolicy(x->rand(HSActionSpace()))
  # the simulator uses the exact dynamics (not known to the belief_updater)
  for i_run in 1:n_runs
    makegif(exact_pomdp, policy, belief_updater, filename=joinpath(@__DIR__, "../renderings/out$i_run.gif"), extra_initial=true, rng=rng, max_steps=100, show_progress=true)
  end
end

struct StraightToTarget <: Policy end

function POMDPs.action(p::StraightToTarget, s::HSState)
  # take the action that moves me closest to goal as a rollout
  best_action = reduce((a1, a2) -> dist_to_pose(apply_action(s.robot_pose, a1), s.robot_target) < dist_to_pose(apply_action(s.robot_pose, a2), s.robot_target) ? a1 : a2, HSActionSpace())
end

function value_lower_bound(mdp::HSMDP, s::HSState, depth::Int)::Float64 # depth is the solver `depth` parameter less the number of timesteps that have already passed (it can be ignored in many cases)
  dist = robot_dist_to_target(s)
  # TODO parameters shoudl be taken from the model
  robot_max_speed = 1
  remaining_step_estimate = dist/robot_max_speed
  # assume that there is only the (discounted) target reward
  return 50.0 * (discount(mdp)^remaining_step_estimate) - remaining_step_estimate * 0.1
end

function test_mdp_solver(n_runs::Int=1)
  rng = MersenneTwister(1)

  mdp_exact = HSMDP(transition_model=PControlledHumanTransition())

  # @requirements_info MCTSSolver() mdp_awgn initialstate(mdp_awgn, rng)

  # for now we run the ...
  # - the planner with some stochastic version of the true dynamics
  # - the simulator with the true dynamics unknown to the planner
  rollout_estimator = RolloutEstimator(StraightToTarget())
  solver = MCTSSolver(estimate_value=rollout_estimator, n_iterations=2000, depth=15, exploration_constant=5.0)
  planner = solve(solver, mdp_exact)
  simulator = HistoryRecorder(rng=rng, max_steps=100)

  for i_run in 1:n_runs
    sim_hist = simulate(simulator, mdp_exact, planner)
    makegif(mdp_exact, sim_hist, filename=joinpath(@__DIR__, "../renderings/out_mcts$i_run.gif"), extra_initial=true, show_progress=true)
    println("Discounted reward: $(discounted_reward(sim_hist))")
  end
end

function demo_mcts_blief_updater(n_runs::Int=1)
  rng = MersenneTwister(7)

  mdp_awgn = HSMDP(transition_model=PControlledHumanAWGNTransition())
  pomdp_awgn = HSPOMDP(sensor=NoisyPositionSensor([0.3, 0.3, 0.3]), mdp=mdp_awgn)

  # blief updater on the pomdp
  belief_updater = SIRParticleFilter(pomdp_awgn, 2000, rng=rng)

  # for now, run the planner on the fully observable version
  rollout_estimator = RolloutEstimator(StraightToTarget())
  solver = MCTSSolver(estimate_value=rollout_estimator, n_iterations=2000, depth=15, exploration_constant=5.0)
  planner = solve(solver, mdp_awgn)
  simulator = HistoryRecorder(rng=rng, max_steps=100, show_progress=true)

  global i_run = 0
  while i_run < n_runs
      sim_hist = simulate(simulator, pomdp_awgn, planner, belief_updater)
      makegif(pomdp_awgn, sim_hist, filename=joinpath(@__DIR__, "../renderings/out_updater_and_mcts$i_run.gif"), extra_initial=true, show_progress=true)
      global i_run += 1
  end

end
