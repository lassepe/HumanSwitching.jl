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
    planning_cpu_time_wrapped = CPUtoq()
    if isnothing(i)
        info = (planning_cpu_time=planning_cpu_time_wrapped,
                prediction_cpu_time=0.0)
    else
        # some policies don't distinguish between prediciton and planning
        # Thus, we fall back to 0.0
        info = merge(i, Dict(:prediction_cpu_time=>get(i, :prediction_cpu_time, 0.0),
                             :planning_cpu_time=>get(i, :planning_cpu_time, planning_cpu_time_wrapped)))
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
    updater_cpu_time = CPUtoq()
    if isnothing(i)
        info = (updater_cpu_time=updater_cpu_time,)
    else
        info = merge(i, Dict(:updater_cpu_time=>updater_cpu_time))
    end
    return bp, info
end

POMDPs.initialize_belief(tu::TimedUpdater, d) = initialize_belief(tu.u, d)
