struct StraightToGoal{P<:HSModel} <: Policy
    problem::P
end

function POMDPs.action(p::StraightToGoal, rp::Pos)
    # take the action that moves me closest to goal as a rollout
    @assert length(actions(problem(p), rp)) == 6

    # NOTE: now that there is the special action, we don't actually need to optimize here.
    #=
    best_action = reduce((a1, a2) -> dist_to_pos(apply_robot_action(rp, a1), robot_goal(problem(p)))
                         < dist_to_pos(apply_robot_action(rp, a2), robot_goal(problem(p))) ?
                         a1 : a2,
                         actions(problem(p), rp))
    =#
    return first(actions(problem(p), rp))
end

POMDPs.action(p::StraightToGoal, e::HSExternalState) = action(p, robot_pos(e))
POMDPs.action(p::StraightToGoal, s::HSState) = action(p, robot_pos(s))
POMDPs.action(p::StraightToGoal, b::AbstractParticleBelief) = action(p, first(particles(b)))

# depth is the solver `depth` parameter less the number of timesteps that have already passed (it can be ignored in many cases)
function free_space_estimate(mdp::HSMDP, s::HSState, steps::Int=0)::Float64
    if isterminal(mdp, s)
        return 0
    end
    rm = reward_model(mdp)
    min_remaining_steps = Int(floor(remaining_step_estimate(mdp, robot_pos(s))))

    reward_estimate::Float64 = 0
    # stage cost
    @assert(min_remaining_steps >= 0)
    if min_remaining_steps > 0
        reward_estimate += sum(rm.living_penalty*(rm.discount_factor^(i-1)) for i in 1:min_remaining_steps)
        # terminal cost for reaching the goal
        reward_estimate += rm.goal_reached_reward*(rm.discount_factor^(min_remaining_steps-1))
    else
        # in this very edge case the goal reached reward must not be
        # discounted since the optimistic remaining step estimate is already 0 but the state is non terminal!
        reward_estimate += rm.goal_reached_reward
    end

    return reward_estimate
end

function free_space_estimate(pomdp::HSPOMDP, s::HSState, b::Any=nothing, ::Int=0)
    return free_space_estimate(mdp(pomdp), s)
end

function free_space_estimate(pomdp::HSPOMDP, b::AbstractParticleBelief)
    # for the free space
    s = first(particles(b))
    return free_space_estimate(mdp(pomdp), s)
end

function get_plan(po::StraightToGoal, belief)
    a, info = action_info(po, belief)
    rp = belief |> particles |> first |> external |> robot_pos
    plan_horizon::Int = 10
    planning_steps = []
    for i in 1:plan_horizon
        ra = POMDPs.action(po, rp)
	rp = apply_robot_action(rp, ra)
	planning_step = (robot_prediction=rp,)
    	push!(planning_steps, planning_step)
    end
    return planning_steps
end



