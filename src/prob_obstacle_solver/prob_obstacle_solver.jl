"""
    decoupled(b::Belief)

Returns the belief over the decoupled states, whose evolution can be predicted independently of the rest.

    decoupled(b::S)

Returns the decoupled part of a state, whose evolution can be predicted independently of the rest.
"""
function decoupled end

decoupled(b) = b
decoupled(pc::ParticleCollection{S}) where S = ParticleCollection([decoupled(p) for p in pc])


"""
    ProbObstacleSolver
"""
@with_kw struct ProbObstacleSolver <: Solver
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
struct ProbObstaclePolicy{B<:BeliefPropagator} <: Policy
    "The solver instance to share relevant hyper parameters"
    sol::ProbObstacleSolver
end

struct POMDPs.solve(sol::ProbObstacleSolver, m::HSPOMDP)
    ProbObstaclePolicy(m, max_search_depth, collision_prob_thresh)
end

function POMDPs.action(po::ProbObstaclePolicy, b::ParticleCollection; debug::Bool=true)
    # 1. Create propagate particles until `sol.max_search_depth` is reached.
    #     - The belief state for each time step is used as open loop belief
    #       predicition for the humans external state.
    predictions::Vector{ParticleCollection} = []
    resize!(predictions, po.max_search_depth + 1)
    predictions[1] = initialize_belief(po.sol.belief_propagator, decoupled(b))
    for i in 2:(max_search_depth + 1)
        predictions[i] = predict(po.sol.belief_propagator, predicitons[i-1])
    end

    # 2. Perform time varying A* on this set of predicitons
    #    -  Challenges:
    #       - robot states won't match exactly. (finite precision)
    throw("Test Error")
end
