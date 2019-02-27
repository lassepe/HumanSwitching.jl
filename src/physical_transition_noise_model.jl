"""
Defining some post transition tranformations. This is used to add noise to the
transition (independent of the HumanBehaviorState)
"""

function apply_physical_transition_noise(ptnm::HSPhysicalTransitionNoiseModel, external::HSExternalState, rng::AbstractRNG)::HSExternalState
  if ptnm isa HSIdentityPTNM
    return external
  elseif ptnm isa HSGaussianNoisePTNM
    # add AWGN to the pose and have small likelyhood of chaning the target
    human_pose_p::Pose = human_pose(external) + rand(rng, MvNormal([0, 0, 0], ptnm.pose_cov))
    robot_pose_p::Pose = robot_pose(external) + rand(rng, MvNormal([0, 0, 0], ptnm.pose_cov))
    return external_state_p = HSExternalState(human_pose_p, robot_pose_p)

    #  # sample a new human behavior with with a small probability
    #  do_resample = rand(rng) < ptnm.model_change_prob
    #  hbs_p = do_resample ? generate_human_behavior(rng, model) : hbs(sp)

    #  return HSState(external=external_state_p,
    #                 hbs=hbs_p)
  else
    @error "Unknown PTNM"
  end
end
