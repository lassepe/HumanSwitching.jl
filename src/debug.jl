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

function random_agent_state(range_x::Array{Float64}=[0., 10.], range_y::Array{Float64}=[0., 10.])::AgentState
  x = (rand() * (range_x[2] - range_x[1])) - range_x[1]
  y = (rand() * (range_y[2] - range_y[1])) - range_y[1]
  phi = rand() * pi
  return AgentState(xy=[x, y], phi=phi)
end

function rendering_test()
  room = RoomRep(width=10, height=10)
  robot_state = random_agent_state()

  human_states = [random_agent_state() for i in 1:5]

  composition = render_scene(room, robot_state, human_states)
  composition |> SVG("display.svg", 14cm, 14cm)

  introspection = introspect(composition)
  introspection |> SVG("introspection.svg", 14cm, 14cm)

  return composition
end

rendering_test()
