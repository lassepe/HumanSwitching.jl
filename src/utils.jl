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

# determines the 2D vector from p_start to p_end
vec_from_to(p_start::Pose, p_end::Pose)::SVector{2} = p_end[1:2] - p_start[1:2]
# computes the 2-norm distance between p1 and p2 (orientation ignored)
dist_to_pose(p1::Pose, p2::Pose)::Float64 = norm(vec_from_to(p1, p2))
# computes the distance between the robot and it's target
robot_dist_to_target(s::HSState)::Float64 = norm(dist_to_pose(s.robot_pose, s.robot_target))
# computes the vector pointing from the human to it's target
human_vec_to_target(s::HSState)::SVector{2} = vec_from_to(s.human_pose, s.human_target)
# computes the distance between the human and it's target
human_dist_to_target(s::HSState)::Float64 = norm(human_vec_to_target(s))

function human_angle_to_target(s::HSState)::Float64
  v = human_vec_to_target(s)
  return atan(v[2], v[1])
end

function rand_pose(r::RoomRep; rng::AbstractRNG=Random.GLOBAL_RNG, forced_orientation::Union{Float64, Nothing}=nothing)::Pose
  x = rand(rng) * r.width
  y = rand(rng) * r.height
  phi = forced_orientation === nothing ? rand(rng) * pi : forced_orientation
  return Pose(x, y, phi)
end
rand_pose(m::HSModel; rng::AbstractRNG=Random.GLOBAL_RNG, forced_orientation::Union{Float64, Nothing}=nothing)::Pose = rand_pose(room(m); rng=rng)

function isinroom(as::Pose, r::RoomRep)
  return  0 <= as.x <= r.width && 0 <= as.y <= r.height
end
