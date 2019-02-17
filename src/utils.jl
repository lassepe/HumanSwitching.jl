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

function Base.isequal(a::HSState, b::HSState)
  isequal(external(a), external(b)) && isequal(hbm(a), hbm(b))
end
function Base.isequal(a::HSExternalState, b::HSExternalState)
  isequal(human_pose(a), human_pose(b)) &&
  isequal(robot_pose(a), robot_pose(b))
end
function Base.isequal(a::HumanBehaviorModel, b::HumanBehaviorModel)
  # TODO refactorState
  isequal(human_target(a), human_target(b))
end

# determines the corner poses of the room
corner_poses(r::RoomRep) = [Pose(x, y, 0) for x in [0.1r.width, 0.9r.width], y in [0.1r.height, 0.9r.height]]

# determines the 2D vector from p_start to p_end
vec_from_to(p_start::Pose, p_end::Pose)::SVector{2} = p_end[1:2] - p_start[1:2]
# computes the 2-norm distance between p1 and p2 (orientation ignored)
dist_to_pose(p1::Pose, p2::Pose)::Float64 = norm(vec_from_to(p1, p2))
# computes the distance between the robot and it's target
robot_dist_to_target(m::HSModel, s::HSState)::Float64 = norm(dist_to_pose(robot_pose(s), robot_target(m)))
# computes the vector pointing from the human to it's target
human_vec_to_target(s::HSState)::SVector{2} = vec_from_to(human_pose(s), human_target(s))
# computes the distance between the human and it's target
human_dist_to_target(s::HSState)::Float64 = norm(human_vec_to_target(s))
# checks if the state currently has a collision between the robot and some other agent
has_collision(m::HSModel, s::HSState)::Bool = dist_to_pose(human_pose(s), robot_pose(s)) < agent_min_distance(m)

human_reached_target(s::HSState)::Bool = human_dist_to_target(s) < 0.2
robot_reached_target(m::HSModel, s::HSState)::Bool = robot_dist_to_target(m, s) < 0.6

function human_angle_to_target(s::HSState)::Float64
  v = human_vec_to_target(s)
  return atan(v[2], v[1])
end

function rand_pose(r::RoomRep, rng::AbstractRNG; forced_orientation::Union{Float64, Nothing}=nothing)::Pose
  x = rand(rng) * r.width
  y = rand(rng) * r.height
  phi = forced_orientation === nothing ? rand(rng) * pi : forced_orientation
  return Pose(x, y, phi)
end
rand_pose(m::HSModel, rng::AbstractRNG; forced_orientation::Union{Float64, Nothing}=nothing)::Pose = rand_pose(room(m))

function rand_state(r::RoomRep, rng::AbstractRNG; known_external_state::Union{HSExternalState, Nothing}=nothing)
  if known_external_state === nothing
    human_init_pose = rand_pose(r, rng)
    robot_init_pose = rand_pose(r, rng; forced_orientation=0.0)
  else
    human_init_pose = human_pose(known_external_state)
    robot_init_pose = robot_pose(known_external_state)
  end

  # the human target is always unknown
  human_target_pose = rand(rng, corner_poses(r))

  # TODO: refactorState
  return HSState(human_init_pose,
                 human_target_pose,
                 robot_init_pose)
end

function isinroom(p::Pose, r::RoomRep)
  return  0 <= p.x <= r.width && 0 <= p.y <= r.height
end
