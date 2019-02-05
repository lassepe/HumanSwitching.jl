"""
room node

Composes a room node for the visualization graph

Fields:
- `rr` a representation of the room
"""
function room_node(rr::RoomRep; fill_color="bisque", stroke_color="black")::Context
  compose(context(), fill(fill_color), stroke(stroke_color),
          rectangle(0, rr.height, rr.width, rr.height))
end
"""
agent_node

Composes an agent node for the visualization graph

Fields:
- `as` the state of the agent to be rendered
- `r` the visual radius of the agent
"""
function agent_node(as::AgentState; r=0.15, fill_color="tomato", stroke_color="black")::Context
  compose(context(), fill(fill_color), stroke(stroke_color),
          (context(), circle(as.x, as.y, r)),
          (context(), line([(as.x, as.y), (as.x+cos(as.phi)*r*2, as.y+sin(as.phi)*r*2)]), linewidth(1)))
end

"""
target_node

Composes a target node (states that agents want to reach) for the visualization graph

Fields:
- `ts` the target state to be visualized
"""
function target_node(as::AgentState; size=0.15, fill_color="deepskyblue", stroke_color="black")::Context
  compose(context(), fill(fill_color), stroke(stroke_color),
          circle(as.x, as.y, size/2))
end

"""
start_target_curve_node

Composes a line from a given start to a given target to associate an agent with it's target

Fields:
- `start_pose` the pose from where the curve starts (defining position and slope of the curve)
- `target` the target position towards which the curve points (end anchor point, only for position)
- `r` the visual radius of the agent
"""
function start_target_curve_node(start_pose::AgentState, target::AgentState; r=0.5, stroke_color="green")::Context
  # the start and end anchor point for the bezier curve
  p_start = [Tuple(start_pose[1:2])]
  p_end = [Tuple(target[1:2])]
  # control point at the tip of the agents facing direction
  c1 = [(start_pose.x+cos(start_pose.phi)*r*2, start_pose.y+sin(start_pose.phi)*r*2)]

  # control point half way way between the agent and the target
  c2_help = (start_pose[1:2] + target[1:2]) / 2
  c2 = [(c2_help[1], c2_help[2])]

  compose(context(), stroke(stroke_color), curve(p_start, c1, c2, p_end))
end

function agent_with_target_node(agent_pose::AgentState, target::AgentState; agent_color="tomato", curve_color="green", target_color="light green")
  # the actual target of the human
  current_target_viz = target_node(target, size=0.2, fill_color=target_color)
  # the current pose of the human
  agent_pose_viz = agent_node(agent_pose, fill_color=agent_color)
  # a connection line between the human and the target
  target_curve_viz = start_target_curve_node(agent_pose, target, stroke_color=curve_color)

  compose(context(), agent_pose_viz, current_target_viz, target_curve_viz)
end

function belief_node(bp::AbstractParticleBelief)
  some_p = rand(particles(bp))

  compose(context(), [agent_with_target_node(p.human_pose, p.human_target, agent_color="light blue", curve_color="light blue", target_color= "white") for p in particles(bp)])
end

"""
render_step_compose

Renders the whole scene based on the HumanSwitching model and a coresponding
POMDP state.

This returns a Compose.Context that can be rendered to a variety of displays.
(e.g. SVG  or Blink)

Fields:

- `m` the model/problem to be rendered (to extact the room size)
- `step` the step to be rendered (containing the state, the belief, etc.)

"""
function render_step_compose(m::HSModel, step::NamedTuple)::Context
  # extract the relevant information from the step
  sp = step[:sp]

  # TODO: MOVE!
  if haskey(step, :bp) && step[:bp] isa AbstractParticleBelief
    belief_viz = belief_node(step[:bp])
  else
    println("Has no AbstractParticleBelief!")
    belief_viz = context()
  end

  # extract the room prepresentation from the problem
  room_rep::RoomRep = room(m)
  # place mirror all children along the middle axis of the unit context
  mirror = context(mirror=Mirror(0, 0.5, 0.5))
  # scale all children to fit into the mirrored unit context
  base_scale = context(0, 0, 1/room_rep.width, 1/room_rep.height)

  # the room background
  room_viz = room_node(room_rep)
  # all targets where humans might go
  potential_targets = corner_states(room_rep)
  potential_targets_viz = [target_node(pt) for pt in potential_targets]
  # the human and it's target
  agent_with_target_viz = agent_with_target_node(sp.human_pose, sp.human_target)

  compose(mirror, (base_scale, agent_with_target_viz, potential_targets_viz..., belief_viz, room_viz))
end

"""
Same as above but rendering directly to an svg
"""
render_step_svg(m::HSModel, step::NamedTuple) = render_step_compose(m, step) |> SVG(14cm, 14cm)
render_step_svg(m::HSModel, step::NamedTuple, filename::String) = render_step_compose(m, step) |> SVG(filename, 14cm, 14cm)
"""
Same as above but rendering directly to a (potentially provided) blink window.
"""
render_step_blink(m::HSModel, step::NamedTuple, win::Blink.Window) = blink!(render_step_compose(m, step), win)
render_step_blink(m::HSModel, step::NamedTuple) = blink!(render_step_compose(m, step))

"""
blink!

Is a workaround to render Compose.jl context to blink windows by:

- first drawing the composition to an SVG object
- then rendering this object in blink
"""
function blink!(c::Context, win::Blink.Window)
  sp = SVG(600px, 600px, false)
  draw(s, c)
  # make sure that blink is used with options async=true and
  # fade=false to make a better animation
  body!(win, s, async=true, fade=false)
end
blink!(c::Context) = blink!(c, Blink.Window())

# Some interface code to use the POMDPGifs package. This basically needs to im
struct HSViz
  m::HSModel
  step::NamedTuple
end

render(m::HSModel, step::NamedTuple) = HSViz(m, step)

function Base.show(io::IO, mime::MIME"image/png", v::HSViz)
  c = render_step_compose(v.m, v.step)
  surface = CairoRGBSurface(800, 800)
  draw(PNG(surface), c)
  write_to_png(surface, io)
end
