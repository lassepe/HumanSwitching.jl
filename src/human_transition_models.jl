"""
Defnining some human transition models. (dynamics according to which human
move)
"""
# Differen human behaviors
function human_transition(hb::HumanPIDBehavior, p::Pose)::Pose
  human_velocity = min(hb.max_speed, dist_to_pose(p, human_target(hb))) #m/s
  vec2target = vec_from_to(p, human_target(hb))
  target_direction = normalize(vec2target)
  current_walk_direction = @SVector [cos(p.phi), sin(p.phi)]
  walk_direction = (target_direction + current_walk_direction)/2
  # new position:
  human_pose_p::Pose = p
  if !any(isnan(i) for i in target_direction)
    xy_p = p[1:2] + walk_direction * human_velocity
    phi_p = atan(walk_direction[2], walk_direction[1])
    human_pose_p = [xy_p..., phi_p]
  end

  return human_pose_p
end

function human_transition(hb::HumanConstantVelocityBehavior, p::Pose)::Pose
  dp = Pose(cos(p.phi), sin(p.phi), 0) * hb.velocity
  return Pose(p + dp)
end
