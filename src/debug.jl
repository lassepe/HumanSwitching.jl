using Pkg

if !haskey(Pkg.installed(), "HumanSwitching")
  # load the environment if not yet done
  jenv = joinpath(dirname(@__FILE__()), "../.")
  Pkg.activate(jenv)
  @info("Activated Environment")
end

using POMDPs
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

function render_state_test()
  hs_pomdp_noisy, rng, a, s = get_test_problem()

  composition = render_scene(hs_pomdp_noisy, s)
  composition |> SVG("display.svg", 14cm, 14cm)
end

function simulate_state_trajectories(n_traj::Int64=1)
  hs_pomdp_noisy, rng, a, s = get_test_problem()
  win = Blink.Window()
  # simulate a simple state trajectory
  @showprogress for i in 1:n_traj
    s = initialstate(hs_pomdp_noisy, rng)
    while HS.human_dist_to_target(s) > 0.1
      s = HS.generate_s(hs_pomdp_noisy, s, a, rng)
      render_scene_blink(hs_pomdp_noisy, s, win)
      sleep(1)
    end
  end
  close(win)
end

function simulate_with_policy()
  belief_updater = NothingUpdater()
  pomdp, rng = get_test_problem()
  policy = observe_only
  history = simulate(HistoryRecorder(max_steps=100), pomdp, policy, belief_updater)
  # now step over the history and render to blink using the usual method
  win = Blink.Window()
  @showprogress for s in eachstep(history, "s")
    render_scene_blink(pomdp, s, win)
    sleep(1)
  end
  close(win)
end

function simulate_with_makegif()
  belief_updater = NothingUpdater()
  pomdp, rng = get_test_problem()
  policy = observe_only

  makegif(pomdp, policy, belief_updater, filename="out.gif ", rng=rng, max_steps=100, show_progress=true)
end
