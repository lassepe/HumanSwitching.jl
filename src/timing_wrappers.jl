"""
TimedPolicy

A thin planner/policy wrapper to add the time to the action info.
"""
struct TimedPolicy{P<:Policy} <: Policy
    p::P
end

unwrap(policy::TimedPolicy) = policy.p
unwrap(policy::Policy) = policy

POMDPs.action(tp::TimedPolicy, x) = action(tp.p, x)

function POMDPModelTools.action_info(tp::TimedPolicy, x; kwargs...)
    planning_time_wrapped = @elapsed begin
        action, i = action_info(tp.p, x; kwargs...)
    end
    if isnothing(i)
        info = (planning_time=planning_time_wrapped,
                prediction_time=0.0)
    else
        # some policies don't distinguish between prediciton and planning
        # Thus, we fall back to 0.0
        info = merge(i, Dict(:prediction_time=>get(i, :prediction_time, 0.0),
                             :planning_time=>get(i, :planning_time, planning_time_wrapped)))
    end
    return action, info
end

"""
TimedUpdater
"""
struct TimedUpdater{U<:Updater} <: Updater
    u::U
end

POMDPs.update(tu::TimedUpdater, b, a, o) = update(tu.u, b, a, o)

function POMDPModelTools.update_info(tu::TimedUpdater, b, a, o)
    updater_time = @elapsed begin
        bp, i = update_info(tu.u, b, a, o)
    end
    if isnothing(i)
        info = (updater_time=updater_time,)
    else
        info = merge(i, Dict(:updater_time=>updater_time))
    end
    return bp, info
end

POMDPs.initialize_belief(tu::TimedUpdater, d) = initialize_belief(tu.u, d)
