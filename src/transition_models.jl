"""
Defining some post transition tranformations. This is used to add noise to the
transition (independent of the HumanBehaviorModel)
"""
struct HSIdentityPTT <: HSPostTransitionTransform end

@with_kw struct HSGaussianNoisePTT <: HSPostTransitionTransform
  pose_cov::Array{Float64, 1} = [0.15, 0.15, 0.01] # the diagonal of the transition noise covariance matrix
  goal_change_prob::Float64 = 0.01 # probability of randomely changing goal
end

function post_transition_transform(model::HSModel, s::HSState, a::HSAction, sp::HSState, rng::AbstractRNG)::HSState
  pttm = post_transition_transform(model)

  if pttm isa HSIdentityPTT
    return sp
  elseif pttm isa HSGaussianNoisePTT
    # add AWGN to the pose and have small likelyhood of chaning the target
    human_pose_p::Pose = human_pose(sp) + rand(rng, MvNormal([0, 0, 0], pttm.pose_cov))
    do_resample = rand(rng) < pttm.goal_change_prob || human_reached_target(sp)
    human_target_p::Pose = do_resample ? rand(rng, corner_poses(room(model))) : human_target(sp)
    robot_pose_p::Pose = robot_pose(sp) + rand(rng, MvNormal([0, 0, 0], pttm.pose_cov))

    return HSState(human_pose_p,
                   human_target_p,
                   robot_pose_p)
  else
    @error "Unknown PTTM"
  end
end

"""
Defnining some human transition models. (dynamics according to which human
move)
"""
@with_kw struct HumanPIDBehavior <: HumanBehaviorModel
  human_target::Pose
  max_speed::Float64 = 0.5
end

function human_transition(hb::HumanPIDBehavior, p::Pose)::Tuple{HumanBehaviorModel, Pose}
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

  return hb, human_pose_p
end

