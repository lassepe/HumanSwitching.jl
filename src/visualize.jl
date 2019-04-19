const hbsColors = Dict(t=>c for (t, c) in zip(InteractiveUtils.subtypes(HumanBehaviorState),
                                              [get(ColorSchemes.deep, i)
                                               for i in range(1/3, stop=2/3, length=length(InteractiveUtils.subtypes(HumanBehaviorState)))]))

function map_to_color(hbs::HumanBehaviorState)
    # for parametric types we need to find this manually
    fallback_key(hbs::HumanBehaviorState) = first(hbs_type for hbs_type in keys(hbsColors) if hbs isa hbs_type)
    color_key = haskey(hbsColors, typeof(hbs)) ? typeof(hbs) : fallback_key(hbs)
    return hbsColors[color_key]
end

map_to_opacity(normalized_weight::Float64) = sqrt(normalized_weight)
map_to_opacity(weight::Float64, weight_sum::Float64) = map_to_opacity(weight/weight_sum)

"""
room node

Composes a room node for the visualization graph

Fields:
- `rr` a representation of the room
"""
function room_node(rr::Room; fill_color="bisque", stroke_color="black")::Context
    compose(context(), fill(fill_color), stroke(stroke_color),
            rectangle(0, rr.height, rr.width, rr.height))
end

"""
pos_node

Composes an agent node for the visualization graph

Fields:
- `as` the state of the agent to be rendered
- `r` the visual radius of the agent
"""
function pos_node(p::Pos; r::Float64=0.15,
                   fill_color="tomato", stroke_color="black", opacity::Float64=1.0)::Context

    center_line(angle::Float64) = line([(p.x-cos(angle)*r*2, p.y-sin(angle)*r*2),
                                        (p.x+cos(angle)*r*2, p.y+sin(angle)*r*2)])
    marker = compose(context(), center_line(pi/4), center_line(-pi/4),linewidth(1))

    compose(context(), fill(fill_color), fillopacity(opacity), stroke(stroke_color), strokeopacity(opacity),
            circle(p.x, p.y, r), marker)
end

"""
goal_node

Composes a goal node (states that agents want to reach) for the visualization graph

Fields:
- `ts` the goal state to be visualized
"""
function goal_node(p::Pos;
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
start_goal_line_node

Composes a line from a given start to a given goal to associate an agent with it's goal

Fields:
- `start_pos` the pos from where the curve starts (defining position and slope of the curve)
- `goal` the goal position towards which the curve points (end anchor point, only for position)
- `r` the visual radius of the agent
"""
function start_goal_line_node(start_pos::Pos, goal::Pos;
                                 r::Float64=0.5,
                                 stroke_color="green", opacity::Float64=1.0)::Context

    # the start and end anchor point for the bezier curve
    p_start = [Tuple(start_pos[1:2])]
    p_end = [Tuple(goal[1:2])]

    compose(context(), stroke(stroke_color), strokeopacity(opacity),
            line([p_start..., p_end...]))
end

function agent_with_goal_node(agent_pos::Pos, goal::Pos;
                                annotation::String="",
                                goal_size::Float64=0.2,
                                external_color="tomato", curve_color="green",
                                goal_color="light green", opacity::Float64=1.0)::Context
    # the actual goal of the human
    current_goal_viz = goal_node(goal,
                                     annotation=annotation,
                                     size=goal_size,
                                     fill_color=goal_color,
                                     opacity=opacity)
    # the current pos of the human
    agent_pos_viz = pos_node(agent_pos,
                               fill_color=external_color, opacity=opacity)
    # a connection line between the human and the goal
    goal_curve_viz = start_goal_line_node(agent_pos, goal,
                                               stroke_color=curve_color, opacity=opacity)

    compose(context(), agent_pos_viz, current_goal_viz, goal_curve_viz)
end

function human_particle_node(human_pos::Pos, hbm::HumanPIDBehavior, hbs::HumanPIDBState;
                             external_color="light blue", internal_color=map_to_color(hbs),
                             annotation::String="", opacity::Float64=1.0)

    return agent_with_goal_node(human_pos,
                                  human_goal(hbm, hbs),
                                  external_color=external_color,
                                  curve_color=internal_color,
                                  annotation=annotation,
                                  goal_color=internal_color,
                                  goal_size=0.4,
                                  opacity=opacity)
end

function human_particle_node(human_pos::Pos, hbm::HumanMultiGoalBoltzmann, hbs::HumanBoltzmannToGoalBState;
                             external_color="light blue", internal_color=map_to_color(hbs),
                             annotation::String="", opacity::Float64=1.0)

    n_prediction_steps=1
    sampled_future_predictions = zeros(Pos, n_prediction_steps)
    sizehint!(sampled_future_predictions, n_prediction_steps)
    for i = 1:n_prediction_steps
        sampled_future_predictions[i] = free_evolution(hbm, hbs, (i > 1 ? sampled_future_predictions[i-1] : human_pos),
                                                       Random.GLOBAL_RNG)
    end

    return compose(context(), [agent_with_goal_node(p,
                                                    hbs.goal,
                                                    annotation=annotation,
                                                    external_color=external_color,
                                                    curve_color=internal_color,
                                                    goal_color=internal_color,
                                                    goal_size=0.4,
                                                    opacity=opacity*map_to_opacity(1.0, 1.0))
                               for p in sampled_future_predictions]...)
end

function human_particle_node(human_pos::Pos, hbm::HumanConstVelBehavior, hbs::HumanConstVelBState;
                             external_color="light green", internal_color=map_to_color(hbs),
                             annotation::String="", opacity::Float64=1.0)

    predicted_future_pos::Pos = human_pos
    # predict future position
    for i in 1:2
        predicted_future_pos = free_evolution(hbs, predicted_future_pos)
    end

    return agent_with_goal_node(human_pos,
                                  predicted_future_pos,
                                  external_color=external_color,
                                  curve_color=internal_color,
                                  goal_color=internal_color,
                                  goal_size=0.4,
                                  opacity=opacity)
end

function human_particle_node(human_pos::Pos, hbm::HumanBoltzmannModel, hbs::HumanBoltzmannBState;
                             external_color="light green", internal_color=map_to_color(hbs),
                             annotation::String="", opacity::Float64=1.0
                            )
    predicted_future_pos::Pos = human_pos
    # predict future position
    n_samples::Int = 1
    sampled_future_predictions = [free_evolution(hbm, hbs, predicted_future_pos, Random.GLOBAL_RNG) for i in 1:n_samples]

    return compose(context(), [agent_with_goal_node(human_pos,
                                                      p,
                                                      external_color=external_color,
                                                      curve_color=internal_color,
                                                      goal_color=internal_color,
                                                      goal_size=0.4,
                                                      opacity=opacity*map_to_opacity(1.0, Float64(n_samples)))
                               for p in sampled_future_predictions]...)
end

plot_compose(args...; kwargs...) = Gadfly.render(plot(args...; kwargs...))

function parameter_histogram_node(values::Array, default_color, bincount::Int, args...)
    return plot_compose(x=length(values) > 1 ? values : Array{Float64, 1}([]),
                        Gadfly.Theme(default_color=default_color),
                        Geom.histogram(bincount=bincount, density=true),
                        args...)
end

function belief_info_node(b::ParticleCollection, m::HSPOMDP)::Context
    hbm = human_behavior_model(m)
    # filling some colums of the data frame for visualization
    human_behavior_states = [hbs(p) for p in particles(b)]

    # histogram of model types
    known_behavior_state_types::Array{Type} = [t for t in InteractiveUtils.subtypes(HumanBehaviorState) if t <: bstate_type(hbm)]

    # an array to collect all plots to be stacked vertically
    vstack_list::Array{Context} = []

    if length(known_behavior_state_types) > 1
        behavior_state_names = [string(t) for t in known_behavior_state_types]
        behavior_state_type_histogram = plot_compose(x=behavior_state_names,
                                                     y=[count(isa.(human_behavior_states, t)) for t in known_behavior_state_types],
                                                     color=behavior_state_names,
                                                     Geom.bar,
                                                     Gadfly.Theme(key_position=:top,
                                                                  key_max_columns=1,
                                                                  discrete_color_scale=Gadfly.Scale.color_discrete_manual([hbsColors[t] for t in known_behavior_state_types]...)
                                                                 ))
        push!(vstack_list, behavior_state_type_histogram)
    end

    # subplots for detailed visualization of the belief distribution within one bstate type
    bstate_subplots::Context = compose(context(), hstack([bstate_subplot_node(t, human_behavior_states, select_submodel(hbm, t))
                                                          for t in known_behavior_state_types]...))
    push!(vstack_list, bstate_subplots)

    return compose(context(), vstack(vstack_list...))
end

function bstate_subplot_node(::Type{HumanPIDBState},
                             unfiltered_hbs_data::Array{<:HumanBehaviorState}, hbm::HumanBehaviorModel)::Context
    # filter data and map to sortable type
    goal_indices = [goal_index(hbs)-1 for hbs in unfiltered_hbs_data if hbs isa HumanPIDBState]

    # compose histogram
    return parameter_histogram_node(goal_indices, hbsColors[HumanPIDBState], 4,
                                    Coord.Cartesian(xmin=0, xmax=length(hbm.potential_goals)),
                                    Guide.title("PID Human: Goal Index Belief"),
                                    Guide.xlabel("Goal Index"))
end

function bstate_subplot_node(::Type{HumanBoltzmannToGoalBState},
                             unfiltered_hbs_data::Array{<:HumanBehaviorState}, hbm::HumanBehaviorModel)::Context
    # filter data
    betas = [hbs.beta for hbs in unfiltered_hbs_data if hbs isa HumanBoltzmannToGoalBState]
    # compose histogram
    return parameter_histogram_node(betas, hbsColors[HumanBoltzmannToGoalBState], 10,
                                    Coord.Cartesian(xmin=0.1, xmax=log(hbm.beta_max)),
                                    Guide.title("Boltzmann Beta Belief"),
                                    Guide.xlabel("beta"),
                                    Gadfly.Scale.x_log)
    return context()
end

function bstate_subplot_node(::Type{HumanConstVelBState},
                             unfiltered_hbs_data::Array{<:HumanBehaviorState}, hbm::HumanBehaviorModel)::Context
    # filter data
    vx = [(hbs.vx) for hbs in unfiltered_hbs_data if hbs isa HumanConstVelBState]
    vy = [(hbs.vy) for hbs in unfiltered_hbs_data if hbs isa HumanConstVelBState]
    # compose histogram
    return hstack(parameter_histogram_node(vx, hbsColors[HumanConstVelBState], 30,
                                    Coord.Cartesian(xmin=-hbm.vel_max, xmax=hbm.vel_max),
                                    Guide.title("Constant Velocity Belief"),
                                    Guide.xlabel("Vel X")),
                 parameter_histogram_node(vy, hbsColors[HumanConstVelBState], 30,
                                    Coord.Cartesian(xmin=-hbm.vel_max, xmax=hbm.vel_max),
                                    Guide.title("Constant Velocity Belief"),
                                    Guide.xlabel("Vel Y")))
end

function bstate_subplot_node(::Type{HumanBoltzmannBState},
                             unfiltered_hbs_data::Array{<:HumanBehaviorState}, hbm::HumanBehaviorModel)::Context
    # filter data
    betas = [hbs.beta for hbs in unfiltered_hbs_data if hbs isa HumanBoltzmannBState]
    # compose histogram
    return parameter_histogram_node(betas, hbsColors[HumanBoltzmannBState], 10,
                                    Coord.Cartesian(xmin=0.1, xmax=log(hbm.beta_max)),
                                    Guide.title("Boltzmann Beta Belief"),
                                    Guide.xlabel("beta"),
                                    Gadfly.Scale.x_log)
end

function belief_node(b::ParticleCollection{H}, m::HSPOMDP) where H <: HSState
    # computing the state belief distribution
    state_belief_counter = Counter{HSState, Float64}()
    hbm = human_behavior_model(m)

    # compute some statistics on the belief
    weight_sum::Float64 = 0
    for (p, w) in weighted_particles(b)
        add(state_belief_counter, p, w)
        weight_sum += w
    end
    @assert(weight_sum > 0)

    max_visualized_particles = 4000
    human_particles = [human_particle_node(human_pos(p), select_submodel(hbm, hbs(p)), hbs(p);
                                           annotation=string(round(state_count/weight_sum, digits=3)),
                                           opacity=map_to_opacity(state_count, weight_sum))
                       for (idx, (p, state_count)) in enumerate(state_belief_counter) if idx < max_visualized_particles]

    robot_particles = [pos_node(robot_pos(p),
                                 fill_color="light green")
                       for (idx, (p, state_count)) in enumerate(state_belief_counter) if idx < max_visualized_particles]

    belief_viz = compose(context(), robot_particles, human_particles)

    return belief_viz
end

function human_prediction_node(b::ParticleCollection{H}, m::HSPOMDP) where H <: HSHumanState
    # computing the state belief distribution
    state_belief_counter = Counter{HSHumanState, Float64}()
    hbm = human_behavior_model(m)

    # compute some statistics on the belief
    weight_sum::Float64 = 0
    for (p, w) in weighted_particles(b)
        add(state_belief_counter, p, w)
        weight_sum += w
    end
    @assert(weight_sum > 0)

    max_visualized_particles = 4000
    human_particles = [human_particle_node(hp, select_submodel(hbm, hbs), hbs;
                                           annotation=string(round(state_count/weight_sum, digits=3)),
                                           opacity=map_to_opacity(state_count, weight_sum))
                       for (idx, ((hp, hbs), state_count)) in enumerate(state_belief_counter) if idx < max_visualized_particles]

    belief_viz = compose(context(), human_particles)

    return belief_viz
end

function reward_node(step::NamedTuple, sim_hist::T) where T<:POMDPHistory
    cumulative_reward_history = cumsum([r for (t, r) in eachstep(sim_hist, ":t, :r") if t <= step.t])
    if !isempty(cumulative_reward_history)
        return plot_compose(x=(1:length(cumulative_reward_history)),
                            y=cumulative_reward_history,
                            Guide.title("Cumulative Reward"),
                            Guide.xlabel("time"),
                            Guide.ylabel("cumulative reward"),
                            Geom.line,
                            Geom.point,
                            Coord.Cartesian(xmin=0, xmax=length(sim_hist)))
    else
        return context()
    end
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
function render_step_compose(m::HSModel, step::NamedTuple, base_aspectratio::Float64, sim_hist,
                             show_info::Bool)::Context
    # extract the relevant information from the step
    sp = step[:sp]

    # extract the room prepresentation from the problem
    room_rep::Room = room(m)
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

    # the human and it's goal
    human_ground_truth_viz = agent_pos_viz = pos_node(human_pos(sp),
                                                      fill_color="tomato", opacity=1.0)
    # the robot and it's goal
    robot_with_goal_viz = agent_with_goal_node(robot_pos(sp), robot_goal(m),
                                               external_color="pink", curve_color="steelblue")

    belief_viz = (haskey(step, :bp) && step[:bp] isa ParticleCollection ?
                  belief_node(step[:bp], m) : context())
    # the info area
    background = compose(context(), rectangle(0, 0, 1, 1), fill("white"))
    info_position_context = (base_aspectratio < 1 ?
                             context(0, 0, 1, 1-base_aspectratio) :
                             context(1/base_aspectratio, 0, 1 - 1/base_aspectratio, 1))
    if show_info
        belief_info_viz = (haskey(step, :bp) && step[:bp] isa ParticleCollection ?
                           belief_info_node(step[:bp], m) : context())
        reward_info_viz = reward_node(step, sim_hist)
        info_stack = [belief_info_viz, reward_info_viz]
        info_viz = compose(info_position_context, vstack(info_stack...), background)
    else
        info_viz = compose(info_position_context, background)
    end

    if haskey(step, :ai) && haskey(step[:ai], :state_sequence)
        planner_state_squence = step[:ai][:state_sequence]
        # We are visualizing based on sp, thus we need to drop the first state
        robot_plan_viz = path_node([ps.rp for ps in planner_state_squence[2:end]])
    else
        robot_plan_viz = context()
    end

compose(context(),
        (context(), info_viz),
        (mirror, (base_scale,
                  robot_with_goal_viz,
                  robot_plan_viz,
                  human_ground_truth_viz,
                  belief_viz,
                  room_viz))
       )
end

function path_node(way_points::AbstractVector{Pos}; fill_color="black", opacity=0.5)
    return compose(context(),
                   [goal_node(wp; fill_color=fill_color, opacity=opacity, size=0.1) for wp in way_points],
                   strokeopacity(opacity),
                   line([(wp[1], wp[2]) for wp in way_points]))
end

function render_plan_compose(m::HSModel, planning_step::NamedTuple, base_aspectratio::Float64)
    # extract the room prepresentation from the problem
    room_rep::Room = room(m)
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

    # the human and it's goal
    human_ground_truth_viz = agent_pos_viz = pos_node(planning_step[:human_pos],
                                                      fill_color="tomato", opacity=1.0)
    # the robot and it's goal
    robot_with_goal_viz =  agent_with_goal_node(planning_step[:robot_pos], robot_goal(m),
                                                external_color="light green", curve_color="steelblue")

    human_prediction_viz = human_prediction_node(planning_step[:bp], m)

    robot_prediction_viz = pos_node(planning_step[:robot_prediction], fill_color="light green", r=0.1, opacity=0.5)

    # the info area
    background = compose(context(), rectangle(0, 0, 1, 1), fill("white"))

compose(context(),
        (mirror, (base_scale,
                  robot_with_goal_viz,
                  human_ground_truth_viz,
                  robot_prediction_viz,
                  human_prediction_viz,
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

# Some interface code to use the POMDPGifs package.
struct HSViz{M<:HSModel, H<:POMDPHistory, NT<:NamedTuple}
    m::M
    step::NT
    sim_hist::H
    show_info::Bool
end

render(m::HSModel, step::NamedTuple; sim_hist=nothing, show_info=false) = HSViz(m, step, sim_hist, show_info)

function Base.show(io::IO, mime::MIME"image/png", v::HSViz)
    frame_dimensions::Tuple{Float64, Float64} = (v.show_info ? 1600 : 800, 800)
    surface = CairoRGBSurface(frame_dimensions...)
    c = render_step_compose(v.m,
                            v.step,
                            frame_dimensions[1]/frame_dimensions[2],
                            v.sim_hist,
                            v.show_info)
    draw(PNG(surface), c)
    write_to_png(surface, io)
end

struct ProbObstaclePlanViz{M<:HSModel, NT<:NamedTuple}
    m::M
    planning_step::NT
end

render_plan(m::HSModel, planning_step::NamedTuple) = ProbObstaclePlanViz(m, planning_step)

function Base.show(io::IO, mime::MIME"image/png", v::ProbObstaclePlanViz)
    frame_dimensions::Tuple{Float64, Float64} = (800, 800)
    surface = CairoRGBSurface(frame_dimensions...)
    c = render_plan_compose(v.m,
                            v.planning_step,
                            frame_dimensions[1]/frame_dimensions[2])
    draw(PNG(surface), c)
    write_to_png(surface, io)
end
