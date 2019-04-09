POMDPSimulators.problem(p::Policy) = p.problem
POMDPSimulators.problem(p::DESPOTPlanner) = p.pomdp
POMDPSimulators.problem(p::TimedPolicy) = problem(p.p)

validation_hash(hist::SimHistory) = string(hash(collect((sp.external.robot_pos,
                                                         sp.external.human_pos)
                                                        for sp in eachstep(hist, "sp"))))

function final_state_type(m::HSModel, hist::SimHistory)
    final_state = last(collect(eachstep(hist, "sp")))
    if issuccess(m, final_state)
        return "success"
    elseif isfailure(m, final_state)
        return "failure"
    else
        return "nonterminal"
    end
end
