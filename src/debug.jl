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
  policy = FunctionPolicy(x->HSAction())
  # the simulator uses the exact dynamics (not known to the belief_updater)
  for i_run in 1:n_runs
    makegif(exact_pomdp, policy, belief_updater, filename=joinpath(@__DIR__, "../renderings/out$i_run.gif"), extra_initial=true, rng=rng, max_steps=100, show_progress=true)
  end
end
