hbsColors = Dict(t=>c for (t, c) in zip(InteractiveUtils.subtypes(HumanBehaviorState),
                                                  [get(ColorSchemes.deep, i)
                                                   for i in range(1/3, stop=2/3, length=length(InteractiveUtils.subtypes(HumanBehaviorState)))]))
map_to_color(hbs::HumanBehaviorState) = hbsColors[typeof(hbs)]

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

function human_particle_node(human_pose::Pose, hbs::HumanPIDBState;
                             external_color="light blue", internal_color=map_to_color(hbs),
                    annotation::String="", opacity::Float64=1.0)

  return agent_with_target_node(human_pose,
                                human_target(hbs),
                                external_color=external_color,
                                curve_color=internal_color,
                                annotation=annotation,
                                target_color=internal_color,
                                target_size=0.4,
                                opacity=opacity)
end

function human_particle_node(human_pose::Pose, hbs::HumanConstVelBState;
                             external_color="light green", internal_color=map_to_color(hbs),
                    annotation::String="", opacity::Float64=1.0)

  predicted_future_pose::Pose = human_pose
  # predict future position
  for i in 1:4
    predicted_future_pose = free_evolution(hbs, predicted_future_pose)
  end

  return agent_with_target_node(human_pose,
                                predicted_future_pose,
                                external_color=external_color,
                                curve_color=internal_color,
                                target_color=internal_color,
                                target_size=0.4,
                                opacity=opacity)
end

plot_compose(args...; kwargs...) = Gadfly.render(plot(args...; kwargs...))

function belief_info_node(b::ParticleCollection, weight_sum::Float64, m::HSPOMDP)::Context
  # filling some colums of the data frame for visualization
  human_behavior_states = [hbs(p) for p in particles(b)]

  behavior_state_types::Array{Type, 1} = [typeof(hbs) for hbs in human_behavior_states]
  velocities::Array{Float64, 1} = [hbs.velocity for hbs in human_behavior_states if hbs isa HumanConstVelBState]
  target_indices::Array{Int, 1} = [target_index(select_submodel(human_behavior_model(m), hbs), hbs.human_target)
                                   for hbs in human_behavior_states if hbs isa HumanPIDBState]

  # histogram of model types
  all_behavior_state_types = InteractiveUtils.subtypes(HumanBehaviorState)
  behavior_state_names = [string(t) for t in all_behavior_state_types]
  behavior_state_type_histogram = plot_compose(x=behavior_state_names,
                                               y=[count(behavior_state_types .== t) for t in all_behavior_state_types],
                                               color=behavior_state_names,
                                               Geom.bar,
                                               Gadfly.Theme(key_position=:top,
                                                            key_max_columns=1,
                                                            discrete_color_scale=Gadfly.Scale.color_discrete_manual([hbsColors[t] for t in all_behavior_state_types]...)
                                                           ))

  # histogram of constant velocity estimate
  velocity_histogram = plot_compose(x=length(velocities) > 1 ? velocities : Array{Float64, 1}([]),
                                    Geom.histogram(bincount=30, density=true))

  target_histogram = plot_compose(x=length(target_indices) > 1 ? target_indices : Array{Float64, 1}([]),
                                  Geom.histogram(bincount=4, density=true))


  background = compose(context(), rectangle(0, 0, 1, 1), fill("white"))

  return compose(context(),
                 vstack(hstack(behavior_state_type_histogram),
                        hstack(velocity_histogram, target_histogram)),
                 background)
end

function belief_node(b::ParticleCollection, m::HSPOMDP)::Tuple{Context, Context}
  # computing the state belief distribution
  state_belief_counter = Counter{HSState, Float64}()

  # compute some statistics on the belief
  weight_sum::Float64 = 0
  for (p, w) in weighted_particles(b)
    add(state_belief_counter, p, w)
    weight_sum += w
  end
  @assert(weight_sum > 0)
  human_particles = [human_particle_node(human_pose(p), hbs(p);
                                         annotation=string(round(state_count/weight_sum, digits=3)),
                                         opacity=sqrt(round(state_count/weight_sum, digits=3)))
                     for (p, state_count) in state_belief_counter]

  robot_particles = [pose_node(robot_pose(p),
                               has_orientation=false,
                               fill_color="light green")
                     for (p, state_count) in state_belief_counter]

  belief_viz = compose(context(), robot_particles, human_particles)
  belief_info_viz = belief_info_node(b, weight_sum, m)

  return belief_viz, belief_info_viz
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
function render_step_compose(m::HSModel, step::NamedTuple, base_aspectratio::Float64)::Context
  # extract the relevant information from the step
  sp = step[:sp]

  # extract the room prepresentation from the problem
  room_rep::RoomRep = room(m)
  # place mirror all children along the middle axis of the unit context
  mirror = context(mirror=Mirror(0, 0.5, 0.5))
  # scale all children to fit into the mirrored unit context
  if base_aspectratio < 1
    base_scale = context(0, 0, 1/room_rep.width, 1/room_rep.height*base_aspectratio)
  else
    base_scale = context(0, 0, 1/room_rep.width/base_aspectratio, 1/room_rep.height)
  end

  # the room background
  room_viz = room_node(room_rep)

  # the human and it's target
  human_ground_truth_viz = human_particle_node(human_pose(sp), hbs(sp);
                                               external_color="tomato", internal_color="green")

  # the robot and it's target
  robot_with_target_viz = agent_with_target_node(robot_pose(sp), robot_target(m),
                                                 has_orientation=false,
                                                 external_color="pink", curve_color="steelblue")

  belief_viz, belief_info_viz = haskey(step, :bp) && step[:bp] isa ParticleCollection ? belief_node(step[:bp], m) : (context(), context())

  if base_aspectratio < 1
    info_viz = compose(context(0, 0, 1, 1-base_aspectratio), belief_info_viz, fill("green"))
  else
    info_viz = compose(context(1/base_aspectratio, 0, 1 - 1/base_aspectratio, 1), belief_info_viz, fill("green"))
  end

  compose(context(),
          (context(), info_viz),
          (mirror, (base_scale,
                    robot_with_target_viz,
                    human_ground_truth_viz,
                    belief_viz,
                    room_viz))
          )
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
  frame_dimensions::Tuple{Float64, Float64} = (1600, 800)
  surface = CairoRGBSurface(frame_dimensions...)
  c = render_step_compose(v.m, v.step, frame_dimensions[1]/frame_dimensions[2])
  draw(PNG(surface), c)
  write_to_png(surface, io)
end
