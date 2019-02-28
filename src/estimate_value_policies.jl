struct StraightToTarget <: Policy
    m::HSModel
end

function POMDPs.action(p::StraightToTarget, s::HSState)
    # take the action that moves me closest to goal as a rollout
    best_action = reduce((a1, a2) -> dist_to_pose(apply_robot_action(robot_pose(s), a1), robot_target(p.m))
                         < dist_to_pose(apply_robot_action(robot_pose(s), a2), robot_target(p.m)) ?
                         a1 : a2,
                         HSActionSpace())
end

function POMDPs.action(p::StraightToTarget, b::AbstractParticleBelief)
    s = mode(b)
    # take the action that moves me closest to goal as a rollout
    best_action = reduce((a1, a2) -> dist_to_pose(apply_robot_action(robot_pose(s), a1), robot_target(p.m))
                         < dist_to_pose(apply_robot_action(robot_pose(s), a2), robot_target(p.m)) ?
                         a1 : a2,
                         HSActionSpace())
end

const robot_max_speed = maximum(a[1] for a in HSActionSpace())

# depth is the solver `depth` parameter less the number of timesteps that have already passed (it can be ignored in many cases)
function free_space_estimate(mdp::HSMDP, s::HSState, depth::Int)::Float64
    rm = reward_model(mdp)
    dist = robot_dist_to_target(mdp, s)
    remaining_step_estimate = div(dist, robot_max_speed)

    # stage cost
    reward_estimate::Float64 = remaining_step_estimate > 0 ?
    sum(rm.living_penalty*(rm.discount_factor^i) for i in 1:remaining_step_estimate) :
    0.0
    # terminal cost for reaching the goal
    reward_estimate += rm.target_reached_reward*(rm.discount_factor^remaining_step_estimate)

    return reward_estimate
end

function free_space_estimate(pomdp::HSPOMDP, s::HSState, b::Any, steps::Int)
    return free_space_estimate(mdp(pomdp), s, steps)
end
