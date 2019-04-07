"""
Definining some human transition models. (dynamics according to which human
move)
"""

function human_transition(hbs::HumanConstVelBState, hbm::HumanConstVelBehavior, m::HSModel,
                          p::Pos, rng::AbstractRNG)
    human_pos_p = free_evolution(hbs, p)
    hbs_p = (!iszero(hbm.vel_resample_sigma) ? HumanConstVelBState(hbs.vx + randn(rng)*hbm.vel_resample_sigma,
                                                                   hbs.vy + randn(rng)*hbm.vel_resample_sigma) :
             hbs)

    return human_pos_p, hbs_p
end

function human_transition(hbs::HumanPIDBState, hbm::HumanPIDBehavior, m::HSModel,
                          p::Pos, rng::AbstractRNG)
    human_pos_p = free_evolution(hbm, hbs, p)

    hbs_p = (dist_to_pos(human_pos_p, human_target(hbm, hbs)) < agent_min_distance(m) ?
             hbs=HumanPIDBState(target_index=next_target_index(hbm, hbs)) : hbs)

    return human_pos_p, hbs_p
end

function human_transition(hbs::HumanBehaviorState, hbm::HumanUniformModelMix, m::HSModel,
                          p::Pos, rng::AbstractRNG)
    # select the corresponding sub model
    hbm_sub = select_submodel(hbm, hbs)
    # propagate state according to this model
    human_pos_p, hbs_sub_prime = human_transition(hbs, hbm_sub, m, p, rng)
    # small likelihood of randomly changing behavior state
    hbs_p = rand(rng) < hbm.bstate_change_likelihood ? rand_hbs(rng, hbm) : hbs

    return human_pos_p, hbs_p
end

function human_transition(hbs::HumanBoltzmannBState, hbm::HumanBoltzmannModel, m::HSModel,
                          p::Pos, rng::AbstractRNG)
    hbs_p = hbs
    if rand(rng) < hbm.epsilon
        hbs_p = rand_hbs(rng, hbm)
    end

    # compute the new external state of the human
    return free_evolution(hbm, hbs, p, rng), hbs_p
end

function human_transition(hbs::HumanLinearToGoalBState, hbm::HumanMultiGoalModel, m::HSModel,
                          p::Pos, rng::AbstractRNG)
    human_pos_p = free_evolution(hbs, hbm.vel_max, p)
    # if close to goal, sample next goal according from generative model
    # representing P(g_{k+1} | g_{k})
    if rand(rng) < hbm.goal_resample_sigma
        hbs_p = rand_hbs(rng, hbm)
    else
        hbs_p = (dist_to_pos(human_pos_p, hbs.goal) < agent_min_distance(m) ?
                 hbs=HumanLinearToGoalBState(hbm.next_goal_generator(hbs.goal, hbm.goals, rng)) : hbs)
    end

    return human_pos_p, hbs_p
end
