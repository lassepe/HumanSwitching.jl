using Pkg

if !haskey(Pkg.installed(), "HumanSwitching")
  # load the environment if not yet done
  jenv = joinpath(dirname(@__FILE__()), "../.")
  Pkg.activate(jenv)
  @info("Activated Environment")
end

using Revise
using HumanSwitching
const HS = HumanSwitching
using Printf
using Blink
using Compose
using Random
using ProgressMeter


function get_test_problem()
  # create some test problem
  hs_pomdp_noisy = HS.HSPOMDP(HS.NoisyPositionSensor([0.1,0.1,0.01]))
  a = HS.HSAction()
  rng = MersenneTwister(42)
  s = initialstate(hs_pomdp_noisy, rng)

  return hs_pomdp_noisy, a, s, rng
end

function render_state_test()
  hs_pomdp_noisy, a, s, rng = get_test_problem()

  composition = render_scene(hs_pomdp_noisy, s)
  composition |> SVG("display.svg", 14cm, 14cm)
end

function simulate_state_trajectories(n_traj::Int64=5)
  hs_pomdp_noisy, a, s, rng = get_test_problem()
  win = Blink.Window()
  # simulate a simple state trajectory
  @showprogress for i in 1:n_traj
    s = initialstate(hs_pomdp_noisy, rng)
    while HS.human_dist_to_target(s) > 0.1
      s = HS.generate_s(hs_pomdp_noisy, s, a, rng)
      composition = render_scene_blink(hs_pomdp_noisy, s, win)
      sleep(1)
    end
  end
  close(win)
end
