struct StraightToGoal{MT<:HSModel} <: Policy
    m::MT
end

function POMDPs.action(p::StraightToGoal, s::HSState)
    # take the action that moves me closest to goal as a rollout
    best_action = reduce((a1, a2) -> dist_to_pos(apply_robot_action(robot_pos(s), a1), robot_goal(p.m))
                         < dist_to_pos(apply_robot_action(robot_pos(s), a2), robot_goal(p.m)) ?
                         a1 : a2,
                         HSActionSpace())
end

function POMDPs.action(p::StraightToGoal, b::AbstractParticleBelief)
    s = first(particles(b))
    # take the action that moves me closest to goal as a rollout
    best_action = reduce((a1, a2) -> dist_to_pos(apply_robot_action(robot_pos(s), a1), robot_goal(p.m))
                         < dist_to_pos(apply_robot_action(robot_pos(s), a2), robot_goal(p.m)) ?
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
    remaining_step_estimate = fld(clamp(robot_dist_to_goal(mdp, s, p=2) - goal_reached_distance(mdp), 0, Inf), robot_max_speed)

    reward_estimate::Float64 = 0
    # stage cost
    @assert(remaining_step_estimate >= 0)
    if remaining_step_estimate > 0
        reward_estimate += sum(rm.living_penalty*(rm.discount_factor^(i-1)) for i in 1:remaining_step_estimate)
        # terminal cost for reaching the goal
        reward_estimate += rm.goal_reached_reward*(rm.discount_factor^(remaining_step_estimate-1))
    else
        # in this very edge case the goal reached reward must not be
        # discounted since the optimistic remaining step estimate is already 0 but the state is non terminal!
        reward_estimate += rm.goal_reached_reward
    end

    return reward_estimate
end

function free_space_estimate(pomdp::HSPOMDP, s::HSState, b::Any, ::Int)
    return free_space_estimate(mdp(pomdp), s)
end

function free_space_estimate(pomdp::HSPOMDP, b::ScenarioBelief)
    # for the free space
    s = first(particles(b))
    return free_space_estimate(mdp(pomdp), s)
end
