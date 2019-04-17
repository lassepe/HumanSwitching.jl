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
struct ProbObstaclePolicy{SOL<:ProbObstacleSolver} <: Policy
    "The solver instance to share relevant hyper parameters"
    sol::SOL
end

# TODO maybe the belier propagator should be constructed here
POMDPs.solve(sol::ProbObstacleSolver, ::POMDP) = ProbObstaclePolicy(sol)

function POMDPs.action(po::ProbObstaclePolicy, b; debug::Bool=true)
    # 1. Create propagate particles until `sol.max_search_depth` is reached.
    #     - The belief state for each time step is used as open loop belief
    #       predicition for the humans external state.
    predictions::Vector{ParticleCollection} = []
    resize!(predictions, po.sol.max_search_depth + 1)
    predictions[1] = decoupled(initialize_belief(po.sol.belief_propagator, b))
    for i in 2:(po.sol.max_search_depth + 1)
        predictions[i] = ParticleCollection(predict(po.sol.belief_propagator, predictions[i-1]))
    end

    dump(predictions)

    # 2. Perform time varying A* on this set of predicitons
    #    -  Challenges:
    #       - robot states won't match exactly. (finite precision)
end
