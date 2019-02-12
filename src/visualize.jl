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
pose_node

Composes an agent node for the visualization graph

Fields:
- `as` the state of the agent to be rendered
- `r` the visual radius of the agent
"""
function pose_node(as::Pose; has_orientation::Bool=true, r::Float64=0.15,
                   fill_color="tomato", stroke_color="black", opacity::Float64=1.0)::Context

  if has_orientation
    marker = compose(context(), line([(as.x, as.y), (as.x+cos(as.phi)*r*2, as.y+sin(as.phi)*r*2)]), linewidth(1))
  else
    center_line(angle::Float64) = line([(as.x-cos(angle)*r*2, as.y-sin(angle)*r*2),
                                      (as.x+cos(angle)*r*2, as.y+sin(angle)*r*2)])
    marker = compose(context(), center_line(pi/4), center_line(-pi/4),linewidth(1))
  end

  compose(context(), fill(fill_color), fillopacity(opacity), stroke(stroke_color), strokeopacity(opacity),
          circle(as.x, as.y, r), marker)
end

"""
target_node

Composes a target node (states that agents want to reach) for the visualization graph

Fields:
- `ts` the target state to be visualized
"""
function target_node(as::Pose; size=0.15, fill_color="deepskyblue", stroke_color="black", opacity::Float64=1.0)::Context
  compose(context(), fill(fill_color), fillopacity(opacity), stroke(stroke_color), strokeopacity(opacity),

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
function start_target_curve_node(start_pose::Pose, target::Pose;
                                 has_orientation::Bool=true, r::Float64=0.5,
                                 stroke_color="green", opacity::Float64=1.0)::Context

  # the start and end anchor point for the bezier curve
  p_start = [Tuple(start_pose[1:2])]
  p_end = [Tuple(target[1:2])]
  # control point at the tip of the agents facing direction
  c1 = [(start_pose.x+cos(start_pose.phi)*r*2, start_pose.y+sin(start_pose.phi)*r*2)]

  # control point half way way between the agent and the target
  c2_help = (start_pose[1:2] + target[1:2]) / 2
  c2 = [(c2_help[1], c2_help[2])]

  compose(context(), stroke(stroke_color), strokeopacity(opacity),
          has_orientation ? curve(p_start, c1, c2, p_end) : line([p_start..., p_end...]))
end

function agent_with_target_node(agent_pose::Pose, target::Pose;
                                target_size::Float64=0.2, has_orientation::Bool=true,
                                agent_color="tomato", curve_color="green",
                                target_color="light green", opacity::Float64=1.0)::Context
  # the actual target of the human
  current_target_viz = target_node(target, size=target_size, fill_color=target_color, opacity=opacity)
  # the current pose of the human
  agent_pose_viz = pose_node(agent_pose,
                             has_orientation=has_orientation,
                             fill_color=agent_color, opacity=opacity)
  # a connection line between the human and the target
  target_curve_viz = start_target_curve_node(agent_pose, target,
                                             has_orientation=has_orientation,
                                             stroke_color=curve_color, opacity=opacity)

  compose(context(), agent_pose_viz, current_target_viz, target_curve_viz)
end

function belief_node(bp::AbstractParticleBelief;
                     render_particle_percentage::Float64=0.1)::Context
  # pick only a few particles to render since otherwise rendering takes ages
  particle_subset = (p for (i, p) in enumerate(particles(bp))
                     if i <= render_particle_percentage*length(particles(bp)))

  human_particles = [agent_with_target_node(p.human_pose,
                                            p.human_target,
                                            agent_color="light blue",
                                            curve_color="gray",
                                            target_color="light blue",
                                            target_size=0.4,
                                            opacity=0.1)
                     for p in particle_subset]

  robot_particles = [pose_node(p.robot_pose,
                               has_orientation=false, # TODO just for checking
                               fill_color="light green",
                               opacity=0.1)
                     for p in particle_subset]

  compose(context(), robot_particles, human_particles)
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

  # extract the room prepresentation from the problem
  room_rep::RoomRep = room(m)
  # place mirror all children along the middle axis of the unit context
  mirror = context(mirror=Mirror(0, 0.5, 0.5))
  # scale all children to fit into the mirrored unit context
  base_scale = context(0, 0, 1/room_rep.width, 1/room_rep.height)

  # the room background
  room_viz = room_node(room_rep)
  # all targets where humans might go
  potential_targets = corner_poses(room_rep)
  potential_targets_viz = [target_node(pt) for pt in potential_targets]

  # the human and it's target
  human_with_target_viz = agent_with_target_node(sp.human_pose, sp.human_target)

  # the robot and it's target
  robot_with_target_viz = agent_with_target_node(sp.robot_pose, sp.robot_target,
                                                 has_orientation=false,
                                                 agent_color="pink", curve_color="steelblue")

  belief_viz = haskey(step, :bp) && step[:bp] isa AbstractParticleBelief ? belief_node(step[:bp]) : context()

  compose(mirror, (base_scale,
                   robot_with_target_viz,
                   human_with_target_viz, potential_targets_viz..., belief_viz,
                   room_viz))
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
  c = render_step_compose(v.m, v.step)
  surface = CairoRGBSurface(800, 800)
  draw(PNG(surface), c)
  write_to_png(surface, io)
end
