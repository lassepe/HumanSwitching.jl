"""
GapCheckingPolicy

Policy wrapper for gap based switching
"""
@with_kw struct GapCheckingPolicy{P<:Policy, M} <: Policy
    smarter_policy::P
    problem::M
    prediction_horizon::Int = 10
end

POMDPs.action(gap_policy::GapCheckingPolicy, b::AbstractParticleBelief) = first(action_info(gap_policy, b))

function POMDPModelTools.action_info(gap_policy::GapCheckingPolicy, b::AbstractParticleBelief)
    # 1. check whether there is a gap

    e0 = external(first(particles(b)))
    rp0 = robot_pos(e0)
    hp0 = human_pos(e0)
    upper_bound_policy = StraightToGoal(gap_policy.problem)

    human_max_step = dt * speed_max(human_behavior_model(gap_policy.problem))
    teb = 2*agent_min_distance(gap_policy.problem)
    FRS_radii = [(teb+t*human_max_step) for t in 0:gap_policy.prediction_horizon]

    is_human_reachable(p::Pos, t::Int) = begin
        human_frs = Circle(hp0, FRS_radii[t+1])
        return contains(human_frs, p)
    end

    # simulate the future using the upper bound policy
    check_gap = () -> begin
        if is_human_reachable(rp0, 0)
            return true
        elseif at_robot_goal(gap_policy.problem, rp0)
            return false
        end

        rp_cur = rp0
        for t in 1:gap_policy.prediction_horizon
            # simulate the action / next waypoint with the upper bound policy
            a = action(upper_bound_policy, rp_cur)
            rp_cur = apply_robot_action(rp_cur, a)
            if is_human_reachable(rp_cur, t)
                return true
            elseif at_robot_goal(gap_policy.problem, rp_cur)
                return false
            end
        end
        return false
    end


    gapchecking_time = @elapsed has_gap = check_gap()

    # there is no gap, we can skip the tedious computation
    if !has_gap
        a, info = action_info(upper_bound_policy, rp0)
	policy_used = upper_bound_policy
    else
    	a, info = action_info(gap_policy.smarter_policy, b)
	policy_used = gap_policy.smarter_policy
    end
    info = (policy_used=policy_used, policy_info=info, FRS_radii=FRS_radii,
            prediction_time=gapchecking_time+(!isnothing(info) ? get(info, :prediction_time, 0.0) : 0.0),
            planning_time=!isnothing(info) ? get(info, :planning_time, nothing) : nothing)
    return a, info
end

function get_plan(po::GapCheckingPolicy, belief)
    a, info = action_info(po, belief)
    steps = get_plan(info.policy_used, belief)
    planning_steps = []
    for (i, step) in enumerate(steps)
	planning_step = (policy_used=info.policy_used, used_plan_step=step, FRS_radius=info.FRS_radii[i])
	push!(planning_steps, planning_step)
    end
    return planning_steps
end
