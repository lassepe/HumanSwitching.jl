"""
Defining some post transition tranformations. This is used to add noise to the
transition (independent of the HumanBehaviorModel)
"""

function post_transition_transform(model::HSModel, s::HSState, a::HSAction, sp::HSState, rng::AbstractRNG)::HSState
  pttm = post_transition_transform(model)

  if pttm isa HSIdentityPTT
    return sp
  elseif pttm isa HSGaussianNoisePTT
    # add AWGN to the pose and have small likelyhood of chaning the target
    human_pose_p::Pose = human_pose(sp) + rand(rng, MvNormal([0, 0, 0], pttm.pose_cov))
    do_resample = rand(rng) < pttm.goal_change_prob
    human_target_p::Pose = do_resample ? rand(rng, corner_poses(room(model))) : human_target(sp)
    robot_pose_p::Pose = robot_pose(sp) + rand(rng, MvNormal([0, 0, 0], pttm.pose_cov))

    return HSState(human_pose_p,
                   human_target_p,
                   robot_pose_p)
  else
    @error "Unknown PTTM"
  end
end
