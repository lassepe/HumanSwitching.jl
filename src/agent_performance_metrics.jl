"""
Some metrics to evaluate the performance of the planner.
"""

@with_kw struct AgentPerformance
    min_dist_to_human::Float64 = 0
    time_to_reach_target::Float64 = Inf
    discounted_reward::Float64 = 0
end

function AgentPerformance(m::HSModel, sim_hist::SimHistory)
    AgentPerformance(min_dist_to_human=min_dist_to_human(sim_hist),
                     time_to_reach_target=time_to_reach_target(m, sim_hist),
                     discounted_reward=discounted_reward(sim_hist))
end

function min_dist_to_human(sim_hist::SimHistory)::Float64
    distances_to_human::Array{Float64, 1} = map(s->dist_to_pos(human_pos(s), robot_pos(s)), eachstep(sim_hist, "sp"))
    return minimum(distances_to_human)
end

function time_to_reach_target(m::HSModel, sim_hist::SimHistory)::Float64
    state_in_collision::Array{Bool, 1} = map(s->has_collision(m, s), eachstep(sim_hist, "sp"))
    return any(state_in_collision) ? Inf : length(state_in_collision)
end
