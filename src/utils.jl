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

human_vec_to_target(s::HSState)::SVector{2} = s.human_target[1:2] - s.human_pose[1:2]
human_dist_to_target(s::HSState)::Float64 = norm(human_vec_to_target(s))

function human_angle_to_target(s::HSState)::Float64
  v = human_vec_to_target(s)
  return atan(v[2], v[1])
end

function rand_pose(r::RoomRep; rng::AbstractRNG=Random.GLOBAL_RNG)::Pose
  x = rand(rng) * r.width
  y = rand(rng) * r.height
  phi = rand(rng) * pi
  return Pose(x, y, phi)
end
rand_pose(m::HSModel; rng::AbstractRNG=Random.GLOBAL_RNG)::Pose = rand_pose(room(m); rng=rng)

function isinroom(as::Pose, r::RoomRep)
  return  0 <= as.x <= r.width && 0 <= as.y <= r.height
end
