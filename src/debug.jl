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
using Compose
using Random

function render_node_test()
  room = RoomRep(width=10, height=10)
  robot_state = rand_astate(room)

  human_states = [rand_astate(room) for i in 1:5]

  composition = render_scene(room, robot_state, human_states)
  composition |> SVG("display.svg", 14cm, 14cm)

  introspection = introspect(composition)
  introspection |> SVG("introspection.svg", 14cm, 14cm)

  return composition
end

function render_state_test()
  # create some test problem
  hs_pomdp_noisy = HS.HSPOMDP(HS.NoisyPositionSensor([0.1,0.1,0.01]))
  a = HS.HSAction()
  rng = MersenneTwister(42)
  s = initialstate(hs_pomdp_noisy, rng)

  composition = render_scene(hs_pomdp_noisy, s)
  composition |> SVG("display.svg", 14cm, 14cm)
end


for i = 1:100
  render_state_test()
  sleep(1)
end
