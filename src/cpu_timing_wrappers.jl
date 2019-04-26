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
    CPUtic()
    action, i = action_info(tp.p, x; kwargs...)
    planner_cpu_time_us = CPUtoq()
    if isnothing(i)
        info = (planner_cpu_time_us=planner_cpu_time_us,
                prediction_cpu_time=0.0)
    else
        # some policies don't distinguish between prediciton and planning
        # Thus, we fall back to 0.0
        prediction_cpu_time = get(i, :prediction_cpu_time, 0.0)
        info = merge(i, Dict(:planner_cpu_time_us=>planner_cpu_time_us,
                             :prediction_cpu_time=>prediction_cpu_time))
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
    CPUtic()
    bp, i = update_info(tu.u, b, a, o)
    updater_cpu_time_us = CPUtoq()
    if isnothing(i)
        info = (updater_cpu_time_us=updater_cpu_time_us,)
    else
        info = merge(i, Dict(:updater_cpu_time_us=>updater_cpu_time_us))
    end
    return bp, info
end

POMDPs.initialize_belief(tu::TimedUpdater, d) = initialize_belief(tu.u, d)
