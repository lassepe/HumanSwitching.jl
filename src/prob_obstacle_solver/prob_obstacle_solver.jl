struct ProbObstacleSearchState
    "The position of the robot"
    rp::Pos
    "The time index over the planning horizon."
    t_idx::Int
end

# implementing the search interface
@with_kw struct ProbObstacleSearchProblem{BP, M} <: SearchProblem{ProbObstacleSearchState}
    "The predicted beliefs"
    belief_predictions::BP
    "The initial state from which where to plan the path"
    start_state::ProbObstacleSearchState
    "A reference to the world model"
    model::M
    "The maximum acceptable collision probability when planning"
    collision_prob_thresh::Float64
    "The maximum depth of search"
    max_search_depth::Int = length(belief_predictions)
end

start_state(p::ProbObstacleSearchProblem) = p.start_state
# a goal state is reached if the robot reached it's target or the serach depth reached the planning horizon
is_goal_state(p::ProbObstacleSearchProblem, s::ProbObstacleSearchState) = at_robot_goal(p.model, s.rp) || s.t_idx == p.max_search_depth

function successors(p::ProbObstacleSearchProblem, s::ProbObstacleSearchState)
    successors::Vector{Tuple{ProbObstacleSearchState, HSAction, Float64}} = []
    # if we have reached the max depth, there we don't return successors
    if s.t_idx == p.max_search_depth
        # this should never happen because a state is said to be a goal state
        # if the max_search_depth is reached
        @assert false
        # TODO remove
        # return successors
    end
    resize!(successors, n_actions(p.model))
    for (i, a) in enumerate(actions(p.model, s.rp))
        # TODO: Apply clipping to discrete values
        # new robot position
        rp_p = apply_robot_action(s.rp, a)
        # the new time index
        t_idx_p = s.t_idx + 1
        s_p = ProbObstacleSearchState(rp_p, t_idx_p)

        # step cost
        b = p.belief_predictions[t_idx_p]
        # TODO: do less naive or at least outosurce integration
        # Monte Carlo integration over particles
        p_share = 1 / n_particles(b)
        # cost estimate over propagated particle blief
        rm = reward_model(p.model)
        if !isinroom(rp_p, room(p.model))
            # if we left the room, we are done
            # TODO: Maybe rather continue here and skip this successor?
            c = -rm.left_room_penalty
        else
            c = -rm.living_penalty
            collision_prob = 0.0
            for (human_pos, _) in particles(b)
                if dist_to_pos(rp_p, human_pos) < agent_min_distance(p.model)
                    collision_prob += p_share
                    c -= p_share * (rm.collision_penalty + rm.dist_to_human_penalty)
                elseif dist_to_pos(rp_p, human_pos) < 2 * agent_min_distance(p.model)
                    c -= p_share * rm.dist_to_human_penalty
                end
                if collision_prob >= p.collision_prob_thresh
                    break
                end
            end
        end
        successors[i] = (s_p, a, c)
    end

    return successors
end

"""
    decoupled(b::Belief)

Returns the belief over the decoupled states, whose evolution can be predicted independently of the rest.

    decoupled(b::S)

Returns the decoupled part of a state, whose evolution can be predicted independently of the rest.
"""
function decoupled end

decoupled(b) = b
decoupled(pc::ParticleCollection{S}) where S = ParticleCollection([decoupled(p) for p in particles(pc)])


# TODO: choose a more generic name. This should rather be a "DecoupledStateSolver"
"""
    ProbObstacleSolver
"""
@with_kw struct ProbObstacleSolver{B<:BeliefPropagator} <: Solver
    "Propagates a given belief into the future"
    belief_propagator::B
    "The depth after which the heuristic is used to approximate the cost to go"
    max_search_depth::Int = 10
    "If the collision probability is higher than this, the the corresponding
    state is considered infasible"
    collision_prob_thresh::Float64 = 1e-3
end

"""
    ProbObstaclePolicy
"""
struct ProbObstaclePolicy{SOL<:ProbObstacleSolver, P<:POMDP} <: Policy
    "The solver instance to share relevant hyper parameters"
    sol::SOL
    "The pomdop to be solved"
    pomdp::P
end

# TODO maybe the belier propagator should be constructed here
POMDPs.solve(sol::ProbObstacleSolver, m::POMDP) = ProbObstaclePolicy(sol, m)

function POMDPs.action(po::ProbObstaclePolicy, b)
    return first(action_info(po, b))
end

function POMDPModelTools.action_info(po::ProbObstaclePolicy, b)
    # 1. Create propagate particles until `sol.max_search_depth` is reached.
    #     - The belief state for each time step is used as open loop belief
    #       predicition for the humans external state.
    belief_predictions::Vector{ParticleCollection} = []
    resize!(belief_predictions, po.sol.max_search_depth + 1)
    belief_predictions[1] = decoupled(initialize_belief(po.sol.belief_propagator, b))
    for i in 2:(po.sol.max_search_depth + 1)
        belief_predictions[i] = ParticleCollection(predict(po.sol.belief_propagator, belief_predictions[i-1]))
    end

    info = (m=po.pomdp,
            belief_predictions=belief_predictions)

    # 2. Perform time varying A* on this set of predicitons
    #    -  Challenges:
    #       - robot states won't match exactly. (finite precision)
    return nothing, info
end
