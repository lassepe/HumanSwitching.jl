# TODO: It would be really nice to share the state type with boltzmann to make
# sharing beliefs a bit easier
struct HumanDeterministicBState <: HumanBehaviorState
    goal::Pos
end

@with_kw struct HumanDeterministicPlanner{NA, TA} <: HumanBehaviorModel
    goals::Array{Pos, 1} = corner_positions(Room())
    obstacles::Array{Circle, 1} = []
    next_goal_generator::Function = uniform_goal_generator
    initial_goal_generator::Function = uniform_goal_generator
    speed_max::Float64 = 1.4
    aspace::SVector{NA, TA} = gen_human_aspace(dist=dt*speed_max)
    goal_reached_distance::Float64 = 1.0
    eps_astar_weight::Float64 = 0.2
end

bstate_type(hbm::HumanDeterministicPlanner) = HumanDeterministicBState
rand_hbs(rng::AbstractRNG, hbm::HumanDeterministicPlanner) = HumanDeterministicBState(hbm.initial_goal_generator(hbm.goals, rng))

"""
    HumanPathSearchProblem

Implementing the GraphSearchLight search problem interface.
"""
struct HumanPathSearchProblem <: SearchProblem{Pos, HumanAction}
    start_state::Pos
    goal_state::Pos
    hbm::HumanDeterministicPlanner
end

GraphSearchLight.start_state(p::HumanPathSearchProblem) = p.start_state
GraphSearchLight.is_goal_state(p::HumanPathSearchProblem, s::Pos) = dist_to_pos(s, p.goal_state) < p.hbm.goal_reached_distance

function GraphSearchLight.successors(p::HumanPathSearchProblem, s::Pos)
    successors::Vector{Tuple{Pos, HumanAction, Int}} = []
    sizehint!(successors, length(p.hbm.aspace))

    for a in p.hbm.aspace
        sp = snap_to_finite_resolution(apply_human_action(s, a))

        # check if this is a legal action
        # TODO: maybe also check whether this action would end outside the room?
        if any([contains(o, sp) for o in p.hbm.obstacles])
            continue
        end
        push!(successors, (sp, a, 1))
    end
    return successors
end

function free_evolution(hbm::HumanDeterministicPlanner, hbs::HumanDeterministicBState, p::Pos)
    # The free evolution is as simple
    planning_problem = HumanPathSearchProblem(p, hbs.goal, hbm)
    # setup the heuristic
    h = (s::Pos) -> dist_to_pos(s, hbs.goal)/(hbm.speed_max * dt)

    # TODO: maybe it would be nice to also seed this with the last plan. This
    # could be as easy as checking whether the (truncated) action sequence
    # would still bring us to the goal.
    aseq::Vector{HumanAction}, sseq::Vector{Pos} = try
        weighted_astar_search(planning_problem, h, hbm.eps_astar_weight)
    catch e
        if !(e isa InfeasibleSearchProblemError)
            rethrow(e)
        else
            @warn("Planner human could not find a path to the goal. Stays put!")
        end
        ([zero(HumanAction)], [p])
    end

    # The human replans at every step. Thus it just needs to return the first
    # state of the sequence
    return first(sseq)
end

function human_transition(hbs::HumanDeterministicBState, hbm::HumanDeterministicPlanner, m::HSModel,
                          p::Pos, rng::AbstractRNG)
    # solve the planning problem
    human_pos_p = free_evolution(hbm, hbs, p)
    # if we are the goal, sample a new goal
    hbs_p = (dist_to_pos(human_pos_p, hbs.goal) < hbm.goal_reached_distance ?
             HumanDeterministicBState(hbm.next_goal_generator(hbs.goal, hbm.goals, rng)) : hbs)

    return human_pos_p, hbs_p
end
