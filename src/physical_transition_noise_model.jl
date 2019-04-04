"""
Defining some post transition tranformations. This is used to add noise to the
transition (independent of the HumanBehaviorState)
"""

function apply_physical_transition_noise(ptnm::HSPhysicalTransitionNoiseModel, external::HSExternalState, rng::AbstractRNG)::HSExternalState
    if ptnm isa HSIdentityPTNM
        return external
    elseif ptnm isa HSGaussianNoisePTNM
        # add AWGN to the pose and have small likelihood of chaning the target
        human_pose_p::Pos = human_pose(external) + rand(rng, MvNormal([0, 0], ptnm.pose_cov))
        robot_pose_p::Pos = robot_pose(external) + rand(rng, MvNormal([0, 0], ptnm.pose_cov))
        return external_state_p = HSExternalState(human_pose_p, robot_pose_p)
    else
        @error "Unknown PTNM"
    end
end
