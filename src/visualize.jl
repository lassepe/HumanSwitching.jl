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
function start_target_curve_node(start_pose::AgentState, target::AgentState; r=0.5, stroke_color="black")::Context
  # the start and end anchor point for the bezier curve
  p_start = [Tuple(start_pose[1:2])]
  p_end = [Tuple(target[1:2])]
  # control point at the tip of the agents facing direction
  c1 = [(start_pose.x+cos(start_pose.phi)*r*2, start_pose.y+sin(start_pose.phi)*r*2)]

  # control point half way way between the agent and the target
  c2_help = (start_pose[1:2] + target[1:2]) / 2
  c2 = [(c2_help[1], c2_help[2])]

  compose(context(), fill("black"), stroke("black"), curve(p_start, c1, c2, p_end))
end

"""
render_scene_compose

Renders the whole scene based on the HumanSwitching model and a coresponding
POMDP state.

This returns a Compose.Context that can be rendered to a variety of displays.
(e.g. SVG  or Blink)

Fields:

- `m` the model/problem to be rendered (to extact the room size)
- `s` the state to be rendered (not agent state but whole POMDP state representation)

"""
function render_scene_compose(m::HSModel, s::HSState)::Context
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
  # the actual target of the human
  current_taret_viz = target_node(s.human_target, size=0.2, fill_color="light green")
  # the current pose of the human
  human_pose_viz = agent_node(s.human_pose, fill_color="tomato")
  # a connection line between the human and the target
  target_curve_viz = start_target_curve_node(s.human_pose, s.human_target)

  composition = compose(mirror, (base_scale, human_pose_viz, current_taret_viz, potential_targets_viz..., target_curve_viz, room_viz))
  return composition
end

"""
Same as above but rendering directly to an svg
"""
render_scene_svg(m::HSModel, s::HSState) = render_scene_compose(m, s) |> SVG(14cm, 14cm)
render_scene_svg(m::HSModel, s::HSState, filename::String) = render_scene_compose(m, s) |> SVG(filename, 14cm, 14cm)
"""
Same as above but rendering directly to a (potentially provided) blink window.
"""
render_scene_blink(m::HSModel, s::HSState, win::Blink.Window) = blink!(render_scene_compose(m, s), win)
render_scene_blink(m::HSModel, s::HSState) = blink!(render_scene_compose(m, s))

"""
blink!

Is a workaround to render Compose.jl context to blink windows by:

- first drawing the composition to an SVG object
- then rendering this object in blink
"""
function blink!(c::Context, win::Blink.Window)
  s = SVG(600px, 600px, false)
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
  c = render_scene_compose(v.m, v.step[:s])
  surface = CairoRGBSurface(800, 800)
  draw(PNG(surface), c)
  write_to_png(surface, io)
end
