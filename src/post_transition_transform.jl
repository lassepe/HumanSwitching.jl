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
    robot_pose_p::Pose = robot_pose(sp) + rand(rng, MvNormal([0, 0, 0], pttm.pose_cov))
    external_state_p = HSExternalState(human_pose_p, robot_pose_p)

    # sample a new human behavior with with a small probability
    do_resample = rand(rng) < pttm.model_change_prob
    hbm_p = do_resample ? generate_human_behavior(rng, model) : hbm(sp)

    return HSState(external=external_state_p,
                   hbm=hbm_p)
  else
    @error "Unknown PTTM"
  end
end
