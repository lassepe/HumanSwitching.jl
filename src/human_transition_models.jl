"""
Defnining some human transition models. (dynamics according to which human
move)
"""

function human_transition(hbs::HumanConstVelBState, hbm::HumanConstVelBehavior, m::HSModel,
                          p::Pose, rng::AbstractRNG)
    human_pose_p = free_evolution(hbs, p)
    hbs_p = HumanConstVelBState(rand(rng, TruncatedNormal(hbs.velocity, hbm.vel_sigma, hbm.vel_min, hbm.vel_max)))
    return human_pose_p, hbs_p
end

function human_transition(hbs::HumanPIDBState, hbm::HumanPIDBehavior, m::HSModel,
                          p::Pose, rng::AbstractRNG)
    human_pose_p = free_evolution(hbs, p)

    hbs_p = (dist_to_pose(human_pose_p, human_target(hbs)) < agent_min_distance(m)
             || rand(rng) < hbm.goal_change_likelihood ?
             rand_hbs(rng, hbm) : hbs)

    return human_pose_p, hbs_p
end

# TODO: all of this could be typed more strongly to improve type stability. Avoid jusing abstract classes!
function human_transition(hbs::HumanBehaviorState, hbm::HumanUniformModelMix, m::HSModel,
                          p::Pose, rng::AbstractRNG)
    # select the corresponding sub model
    hbm_sub = select_submodel(hbm, hbs)
    # propagate state according to this model
    human_pose_p, hbs_sub_prime = human_transition(hbs, hbm_sub, m, p, rng)
    # small likelihood of randomly changing behavior state
    hbs_p = rand(rng) < hbm.bstate_change_likelihood ? rand_hbs(rng, hbm) : hbs

    return human_pose_p, hbs_p
end

function human_transition(hbs::HumanBoltzmannBState, hbm::HumanBoltzmannModel, m::HSModel,
                          p::Pose, rng::AbstractRNG)
    hbs_p = hbs
    if !iszero(hbm.epsilon) && (rand(rng) <= hbm.epsilon)
        hbs_p = rand_hbs(rng, hbm)
    end

    # compute the new external state of the human
    return free_evolution(hbm, hbs, p, rng), hbs_p
end
