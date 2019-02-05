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
  hs_pomdp_noisy = HS.HSPOMDP(HS.NoisyPositionSensor([0.1,0.1,0.01]))
  a = HS.HSAction()
  rng = MersenneTwister(42)
  s = initialstate(hs_pomdp_noisy, rng)

  return hs_pomdp_noisy, rng, a, s
end

function simulate_with_policy(n_runs=1)
  belief_updater = NothingUpdater()
  pomdp, rng = get_test_problem()
  policy = FunctionPolicy(x->HSAction())
  # now step over the history and render to blink using the usual method
  win = Blink.Window()

  @showprogress for i in 1:n_runs
    history = simulate(HistoryRecorder(max_steps=100), pomdp, policy, belief_updater)
    for s in eachstep(history, "s")
      render_scene_blink(pomdp, s, win)
      sleep(0.2)
    end
  end

  close(win)
end

function test_belief_updater()
  pomdp, rng = get_test_problem()
  belief_updater = SIRParticleFilter(pomdp, 1000, rng=rng)
  policy = FunctionPolicy(x->HSAction())

  makegif(pomdp, policy, belief_updater, filename="out.gif", rng=rng, max_steps=100, show_progress=true)
end
