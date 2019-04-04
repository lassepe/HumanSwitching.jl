"""
Defining some post transition tranformations. This is used to add noise to the
transition (independent of the HumanBehaviorState)
"""

function apply_physical_transition_noise(ptnm::HSPhysicalTransitionNoiseModel, external::HSExternalState, rng::AbstractRNG)::HSExternalState
    if ptnm isa HSIdentityPTNM
        return external
    elseif ptnm isa HSGaussianNoisePTNM
        # add AWGN to the pos and have small likelihood of chaning the target
        human_pos_p::Pos = human_pos(external) + rand(rng, MvNormal([0, 0], ptnm.pos_cov))
        robot_pos_p::Pos = robot_pos(external) + rand(rng, MvNormal([0, 0], ptnm.pos_cov))
        return external_state_p = HSExternalState(human_pos_p, robot_pos_p)
    else
        @error "Unknown PTNM"
    end
end
