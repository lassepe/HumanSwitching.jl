"""
Defnining some human transition models. (dynamics according to which human
move)
"""
# Differen human behaviors
function human_transition(hbm::HumanPIDBehavior, p::Pose, m::HSModel, rng::AbstractRNG)::Tuple{Pose, HumanPIDBehavior}
  human_velocity = min(hbm.max_speed, dist_to_pose(p, human_target(hbm))) #m/s
  vec2target = vec_from_to(p, human_target(hbm))
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

  hbm_p = (dist_to_pose(human_pose_p, human_target(hbm)) < 0.5 ?
           generate_human_behavior(rng, HumanPIDBehavior, m) : hbm)

  return human_pose_p, hbm_p
end

function human_transition(hbm::HumanConstantVelocityBehavior, p::Pose, m::HSModel, rng::AbstractRNG)::Tuple{Pose, HumanConstantVelocityBehavior}
  dp = Pose(cos(p.phi), sin(p.phi), 0) * hbm.velocity
  return Pose(p + dp), hbm
end
