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
function pose_node(p::Pose; has_orientation::Bool=true, r::Float64=0.15,
                   fill_color="tomato", stroke_color="black", opacity::Float64=1.0)::Context

  if has_orientation
    marker = compose(context(), line([(p.x, p.y), (p.x+cos(p.phi)*r*2, p.y+sin(p.phi)*r*2)]), linewidth(1))
  else
    center_line(angle::Float64) = line([(p.x-cos(angle)*r*2, p.y-sin(angle)*r*2),
                                      (p.x+cos(angle)*r*2, p.y+sin(angle)*r*2)])
    marker = compose(context(), center_line(pi/4), center_line(-pi/4),linewidth(1))
  end

  compose(context(), fill(fill_color), fillopacity(opacity), stroke(stroke_color), strokeopacity(opacity),
          circle(p.x, p.y, r), marker)
end

"""
target_node

Composes a target node (states that agents want to reach) for the visualization graph

Fields:
- `ts` the target state to be visualized
"""
function target_node(p::Pose;
                     annotation::String="",
                     size=0.15, fill_color="deepskyblue",
                     stroke_color="black",
                     opacity::Float64=1.0)::Context
  compose(context(),
         (context(), fill(fill_color), fillopacity(opacity), stroke(stroke_color), strokeopacity(opacity),
          circle(p.x, p.y, size/2)),
         (context(), text(p.x+size/2, p.y+size/2, annotation), fill("black")))
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
                                annotation::String="",
                                target_size::Float64=0.2, has_orientation::Bool=true,
                                external_color="tomato", curve_color="green",
                                target_color="light green", opacity::Float64=1.0)::Context
  # the actual target of the human
  current_target_viz = target_node(target,
                                   annotation=annotation,
                                   size=target_size,
                                   fill_color=target_color,
                                   opacity=opacity)
  # the current pose of the human
  agent_pose_viz = pose_node(agent_pose,
                             has_orientation=has_orientation,
                             fill_color=external_color, opacity=opacity)
  # a connection line between the human and the target
  target_curve_viz = start_target_curve_node(agent_pose, target,
                                             has_orientation=has_orientation,
                                             stroke_color=curve_color, opacity=opacity)

  compose(context(), agent_pose_viz, current_target_viz, target_curve_viz)
end

function human_particle_node(human_pose::Pose, hbm::HumanPIDBehavior;
                    external_color="light blue", internal_color="grey",
                    annotation::String="", opacity::Float64=1.0)

  return agent_with_target_node(human_pose,
                                human_target(hbm),
                                external_color=external_color,
                                curve_color=internal_color,
                                annotation=annotation,
                                target_color=internal_color,
                                target_size=0.4,
                                opacity=opacity)
end

function human_particle_node(human_pose::Pose, hbm::HumanConstantVelocityBehavior;
                    external_color="light green", internal_color="blue",
                    annotation::String="", opacity::Float64=1.0)

  dp::Pose = Pose(cos(human_pose.phi), sin(human_pose.phi), 0) * hbm.velocity
  next_step_target::Pose = human_pose + dp

  return agent_with_target_node(human_pose,
                                next_step_target,
                                external_color=external_color,
                                curve_color=internal_color,
                                target_color=internal_color,
                                target_size=0.4,
                                opacity=opacity)
end

# TODO: Broken on purose. Continue here.
struct DiscreteDistribution
  d = Dict()

function belief_node(bp::AbstractParticleBelief)::Context
  # computing the state belief distribution
  state_belief_dict = Dict()
  model_belief_dict = Dict()

  # compute some statistics on the belief
  weight_sum::Float64 = 0
  for (p, w) in weighted_particles(bp)
    # TODO: Move this into a type that handles this implicitly
    if !haskey(state_belief_dict, p)
      state_belief_dict[p] = 0
    end
    state_belief_dict[p] += w

    if !haskey(model_belief_dict, typeof(hbm(p)))
      state_belief_dict[typeof(hbm(p))] = 0
    end
    state_belief_dict[typeof(hbm(p))] += w


    weight_sum += w
  end
  @assert(weight_sum > 0)

  human_particles = [human_particle_node(human_pose(p), hbm(p);
                                         annotation=string(round(state_count/weight_sum, digits=3)),
                                         opacity=sqrt(round(state_count/weight_sum, digits=3)))
                     for (p, state_count) in state_belief_dict]

  robot_particles = [pose_node(robot_pose(p),
                               has_orientation=false, # TODO just for checking
                               fill_color="light green")
                     for (p, state_count) in state_belief_dict]

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
  human_ground_truth_viz = human_particle_node(human_pose(sp), hbm(sp);
                                               external_color="tomato", internal_color="green")

  # the robot and it's target
  robot_with_target_viz = agent_with_target_node(robot_pose(sp), robot_target(m),
                                                 has_orientation=false,
                                                 external_color="pink", curve_color="steelblue")

  belief_viz = haskey(step, :bp) && step[:bp] isa AbstractParticleBelief ? belief_node(step[:bp]) : context()

  compose(mirror, (base_scale,
                   robot_with_target_viz,
                   human_ground_truth_viz, potential_targets_viz..., belief_viz,
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
function blink!(c::Context, win::Blink.Window = Blink.Window())
  s = SVG(600px, 600px, false)
  draw(s, c)
  # make sure that blink is used with options async=true and
  # fade=false to make a better animation
  body!(win, s, async=true, fade=false)
end

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
