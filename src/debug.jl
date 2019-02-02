using Pkg

if !haskey(Pkg.installed(), "HumanSwitching")
  # load the environment if not yet done
  jenv = joinpath(dirname(@__FILE__()), "../.")
  Pkg.activate(jenv)
  @info("Activated Environment")
end

using Revise
using HumanSwitching
using Printf

using Compose

function rendering_test()
  room = RoomRep(width=10, height=10)
  robot_state = rand_astate(room)

  human_states = [rand_astate(room) for i in 1:5]

  composition = render_scene(room, robot_state, human_states)
  composition |> SVG("display.svg", 14cm, 14cm)

  introspection = introspect(composition)
  introspection |> SVG("introspection.svg", 14cm, 14cm)

  return composition
end

for i in 1:100
  rendering_test()
  sleep(1)
end
