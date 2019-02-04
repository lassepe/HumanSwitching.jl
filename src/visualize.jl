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
          (context(), circle(as.xy[1], as.xy[2], r)),
          (context(), line([(as.xy[1], as.xy[2]), (as.xy[1]+cos(as.phi)*r*2, as.xy[2]+sin(as.phi)*r*2)]), linewidth(1)))
end

"""
target_node

Composes a target node (states that agents want to reach) for the visualization graph

Fields:
- `ts` the target state to be visualized
"""
function target_node(ts::AgentState; size=0.15, fill_color="deepskyblue", stroke_color="black")::Context
  compose(context(), fill(fill_color), stroke(stroke_color),
          circle(ts.xy[1], ts.xy[ 2], size/2))
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

  room_viz = room_node(room_rep)
  human_pose_viz = agent_node(s.human_pose, fill_color="tomato")
  human_target_viz = target_node(s.human_target)

  p_start = [Tuple(s.human_pose.xy)]
  p_end = [Tuple(s.human_target.xy)]

  # TODO MOVE! This is really ugly!
  as = s.human_pose
  at = s.human_target
  r = 0.5
  c1 = [(as.xy[1]+cos(as.phi)*r*2, as.xy[2]+sin(as.phi)*r*2)]
  c1a = (as.xy + at.xy) / 2
  c2 = [(c1a[1], c1a[2])]
  line_to_target = compose(context(), fill("black"), stroke("black"), curve(p_start, c1, c2, p_end))
  composition = compose(mirror, (base_scale, human_pose_viz, human_target_viz, line_to_target, room_viz))
  return composition
end

"""
Same as above but rendering directly to an svg
"""
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

function Base.show(io::IO, mime::Union{MIME"text/html", MIME"image/svg+xml"}, v::HSViz)
  c = render_scene_compose(v.m, v.step[:s])
  surface = Cairo.CairoSVGSurface(io, 800, 800)
  draw(SVG(surface), c)
  finish(surface)
end

function Base.show(io::IO, mime::MIME"image/png", v::HSViz)
  c = render_scene_compose(v.m, v.step[:s])
  surface = Cairo.CairoRGBSurface(800, 800)
  draw(PNG(surface), c)
  Cairo.write_to_png(surface, io)
end
