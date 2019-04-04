"""
Definining some human transition models. (dynamics according to which human
move)
"""

function human_transition(hbs::HumanConstVelBState, hbm::HumanConstVelBehavior, m::HSModel,
                          p::Pos, rng::AbstractRNG)
    human_pose_p = free_evolution(hbs, p)
    hbs_p = (!iszero(hbm.vel_resample_sigma) ? HumanConstVelBState(hbs.vx + randn(rng)*hbm.vel_resample_sigma,
                                                                   hbs.vy + randn(rng)*hbm.vel_resample_sigma) :
             hbs)

    return human_pose_p, hbs_p
end

function human_transition(hbs::HumanPIDBState, hbm::HumanPIDBehavior, m::HSModel,
                          p::Pos, rng::AbstractRNG)
    human_pose_p = free_evolution(hbm, hbs, p)

    hbs_p = (dist_to_pose(human_pose_p, human_target(hbm, hbs)) < agent_min_distance(m) ?
             hbs=HumanPIDBState(target_index=next_target_index(hbm, hbs)) : hbs)

    return human_pose_p, hbs_p
end

# TODO: all of this could be typed more strongly to improve type stability. Avoid jusing abstract classes!
function human_transition(hbs::HumanBehaviorState, hbm::HumanUniformModelMix, m::HSModel,
                          p::Pos, rng::AbstractRNG)
    # select the corresponding sub model
    hbm_sub = select_submodel(hbm, hbs)
    # propagate state according to this model
    human_pose_p, hbs_sub_prime = human_transition(hbs, hbm_sub, m, p, rng)
    # small likelihood of randomly changing behavior state
    hbs_p = rand(rng) < hbm.bstate_change_likelihood ? rand_hbs(rng, hbm) : hbs

    return human_pose_p, hbs_p
end

function human_transition(hbs::HumanBoltzmannBState, hbm::HumanBoltzmannModel, m::HSModel,
                          p::Pos, rng::AbstractRNG)
    hbs_p = hbs
    if rand(rng) <= hbm.epsilon
        hbs_p = rand_hbs(rng, hbm)
    end

    # compute the new external state of the human
    return free_evolution(hbm, hbs, p, rng), hbs_p
end
