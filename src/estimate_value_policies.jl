struct StraightToTarget{MT<:HSModel} <: Policy
    m::MT
end

function POMDPs.action(p::StraightToTarget, s::HSState)
    # take the action that moves me closest to goal as a rollout
    best_action = reduce((a1, a2) -> dist_to_pos(apply_robot_action(robot_pos(s), a1), robot_target(p.m))
                         < dist_to_pos(apply_robot_action(robot_pos(s), a2), robot_target(p.m)) ?
                         a1 : a2,
                         HSActionSpace())
end

function POMDPs.action(p::StraightToTarget, b::AbstractParticleBelief)
    s = first(particles(b))
    # take the action that moves me closest to goal as a rollout
    best_action = reduce((a1, a2) -> dist_to_pos(apply_robot_action(robot_pos(s), a1), robot_target(p.m))
                         < dist_to_pos(apply_robot_action(robot_pos(s), a2), robot_target(p.m)) ?
                         a1 : a2,
                         HSActionSpace())
end

const robot_max_speed = maximum(a[1] for a in HSActionSpace())

# depth is the solver `depth` parameter less the number of timesteps that have already passed (it can be ignored in many cases)
function free_space_estimate(mdp::HSMDP, s::HSState, steps::Int=0)::Float64
    if isterminal(mdp, s)
        return 0
    end
    rm = reward_model(mdp)
    # TODO: THIS MUST CONSIDER THE MARGIN THAT YOU CAN BE AWAY FROM THE TARGWT!!!!!!
    dist = robot_dist_to_target(mdp, s, p=1)
    remaining_step_estimate = fld(dist, robot_max_speed)

    reward_estimate::Float64 = 0
    # stage cost
    @assert(remaining_step_estimate > 0)
    if remaining_step_estimate > 0
        reward_estimate += sum(rm.living_penalty*(rm.discount_factor^(i-1)) for i in 1:remaining_step_estimate)
        # terminal cost for reaching the goal
        reward_estimate += rm.target_reached_reward*(rm.discount_factor^(remaining_step_estimate-1))
    end

    return reward_estimate + 3
end

function free_space_estimate(pomdp::HSPOMDP, s::HSState, b::Any, ::Int)
    return free_space_estimate(mdp(pomdp), s)
end

function free_space_estimate(pomdp::HSPOMDP, b::ScenarioBelief)
    # for the free space
    s = first(particles(b))
    return free_space_estimate(mdp(pomdp), s)
end
