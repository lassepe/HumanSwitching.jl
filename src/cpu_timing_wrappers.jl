"""
TimedPolicy

A thin planner/policy wrapper to add the time to the action info.
"""
struct TimedPolicy{P<:Policy} <: Policy
    p::P
end

POMDPs.action(tp::TimedPolicy, x) = action(tp.p, x)

function POMDPModelTools.action_info(tp::TimedPolicy, x; kwargs...)
    CPUtic()
    action, info = action_info(tp.p, x; kwargs...)
    info[:planner_cpu_time_us] = CPUtoq()
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
        info = Dict(:updater_cpu_time_us=>updater_cpu_time_us)
    else
        i[:updater_cpu_time_us] = updater_cpu_time_us
        info = i
    end
    return bp, info
end

POMDPs.initialize_belief(tu::TimedUpdater, d) = initialize_belief(tu.u, d)
