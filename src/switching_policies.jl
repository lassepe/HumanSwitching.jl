"""
GapCheckingPolicy

Policy wrapper for gap based switching
"""
@with_kw struct GapCheckingPolicy{P<:Policy, M} <: Policy
    smarter_policy::P
    problem::M
    prediction_horizon::Int = 10
end

function POMDPs.action(gap_policy::GapCheckingPolicy, b::AbstractParticleBelief)
    # 1. check whether there is a gap

    e0 = external(first(particles(b)))
    rp0 = robot_pos(e0)
    hp0 = human_pos(e0)
    upper_bound_policy = StraightToGoal(gap_policy.problem)

    is_human_reachable(p::Pos, t::Int) = begin
        teb = agent_min_distance(gap_policy.problem)
        human_max_step = dt * vel_max(human_behavior_model(gap_policy.problem))
        human_frs = Circle(hp0, teb + t * human_max_step)
        return contains(human_frs, p)
    end

    # simulate the future using the upper bound policy
    has_gap = () -> begin
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

    # there is no gap, we can skip the tedious computation
    if !has_gap()
        return action(upper_bound_policy, rp0)
    end

    return action(gap_policy.smarter_policy, b)
end
