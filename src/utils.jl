"""
normalized_angle_diff

normalizes an angle difference to the range between -pi and pi (always pointing
back along the shortest way). (Distance norm on a circular quantity)
"""
function normalized_angle_diff(angle_diff::Float64)::Float64
    phi::Float64 = angle_diff % (2pi)

    return if phi > pi
        phi - 2pi
    elseif phi < -pi
        2pi + phi
    else
        phi
    end
end

clip_to_finite_resolution(p::Pos, digits::Int=5) = Pos(round(p.x, digits=digits), round(p.y, digits=digits))

# modifying copy constructors for immutable types
construct_with(x, p; type_hint=typeof(x)) = type_hint(((f == p.first ? p.second : getfield(x, f)) for f in fieldnames(typeof(x)))...)
construct_with(x, ps...; kwargs...) = reduce((x, p) -> construct_with(x, p; kwargs...), ps, init=x)

@with_kw struct Counter{TK, TV}
    d::Dict{TK, TV} = Dict()
end

function add(c::Counter, key::Any, val::Float64)
    if !haskey(c.d, key)
        c.d[key] = 0.0
    end
    c.d[key] += val
end

Base.getindex(c::Counter, key::Any) = haskey(c.d, key) ? getindex(c.d, key) : 0.0
Base.iterate(c::Counter) = Base.iterate(c.d)
Base.iterate(c::Counter, idx::Int64) = Base.iterate(c.d, idx::Int64)
Base.length(c::Counter) = Base.length(c.d)

function Base.isequal(a::HSState, b::HSState)
    isequal(external(a), external(b)) && isequal(hbs(a), hbs(b))
end

function Base.isequal(a::HSExternalState, b::HSExternalState)
    isequal(human_pos(a), human_pos(b)) &&
    isequal(robot_pos(a), robot_pos(b))
end

# determines the corner positions of the room
function corner_positions(r::Room, relative_margin::Float64=0.25)
    @assert 0 < relative_margin < 0.5
    return vec([Pos(x, y)
                for x in [relative_margin*r.width, (1-relative_margin)*r.width],
                y in [relative_margin*r.height, (1-relative_margin)*r.height]])
end

dist_to_wall(p::Pos, room::Room) = minimum([p.x, room.width-p.x, p.y, room.height-p.y])

# determines the 2D vector from p_start to p_end
vec_from_to(p_start::Pos, p_end::Pos) = SVector(p_end.x - p_start.x, p_end.y - p_start.y)
# computes the 2-norm distance between p1 and p2
dist_to_pos(p1::Pos, p2::Pos; p=2)::Float64 = norm(vec_from_to(p1, p2), p)
# computes the distance between the robot and it's goal
robot_dist_to_goal(m::HSModel, s::HSState; p=2)::Float64 = dist_to_pos(robot_pos(s), robot_goal(m), p=p)
# checks if the state currently has a collision between the robot and some other agent
has_collision(m::HSModel, s::HSState)::Bool = dist_to_pos(human_pos(s), robot_pos(s)) < agent_min_distance(m)
# check if the state is a failure terminal state
isfailure(m::HSModel, s::HSState)::Bool = has_collision(m, s) || !isinroom(robot_pos(s), room(m))
# check if the state is a success success terminal state
issuccess(m::HSModel, s::HSState)::Bool = !isfailure(m, s) && robot_reached_goal(m, s)

robot_reached_goal(m::HSModel, s::HSState)::Bool = robot_dist_to_goal(m, s) < goal_reached_distance(m)
at_robot_goal(m::HSModel, p::Pos) = dist_to_pos(robot_goal(m), p) < goal_reached_distance(m)

function rand_pos(r::Room, rng::AbstractRNG)::Pos
    x = rand(rng) * r.width
    y = rand(rng) * r.height
    return Pos(x, y)
end
rand_pos(m::HSModel, rng::AbstractRNG)::Pos = rand_pos(room(m))

function rand_external_state(r::Room, rng::AbstractRNG)
    human_pos = rand_pos(r, rng)
    robot_pos = rand_pos(r, rng)
    return HSExternalState(human_pos, robot_pos)
end

function rand_state(m::HSModel, rng::AbstractRNG; known_external_state::Union{HSExternalState, Nothing}=nothing)
    # generate external state
    external_state = known_external_state === nothing ? rand_external_state(room(m), rng) : known_external_state

    # generate human internal state
    hbs = rand_hbs(rng, human_behavior_model(m))

    return HSState(external=external_state, hbs=hbs)
end

function isinroom(p::Pos, r::Room)
    return  0 <= p.x <= r.width && 0 <= p.y <= r.height
end
