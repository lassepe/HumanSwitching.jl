struct ProbObstacleSearchState
    "The position of the robot"
    rp::Pos
    "The time index over the planning horizon."
    t_idx::Int
end

# implementing the search interface
@with_kw struct ProbObstacleSearchProblem{POT, M} <: SearchProblem{ProbObstacleSearchState, HSAction}
    "The predicted beliefs"
    prob_obstacle_trees::POT
    "The initial state from which where to plan the path"
    start_state::ProbObstacleSearchState
    "A reference to the world model"
    model::M
    "The maximum acceptable collision probability when planning"
    collision_prob_thresh::Float64
    "The maximum depth of search"
    max_search_depth::Int = length(prob_obstacle_trees)
end

GraphSearchZero.start_state(p::ProbObstacleSearchProblem) = p.start_state
# a goal state is reached if the robot reached it's target or the serach depth reached the planning horizon
GraphSearchZero.is_goal_state(p::ProbObstacleSearchProblem, s::ProbObstacleSearchState) = at_robot_goal(p.model, s.rp) || s.t_idx == p.max_search_depth

function GraphSearchZero.successors(p::ProbObstacleSearchProblem, s::ProbObstacleSearchState)
    successors::Vector{Tuple{ProbObstacleSearchState, HSAction, Float64}} = []
    # if we have reached the max depth, we don't return successors
    # this should never happen because a state is said to be a goal state
    # if the max_search_depth is reached
    @assert s.t_idx != p.max_search_depth

    sizehint!(successors, n_actions(p.model))
    for a in actions(p.model, s.rp)
        # new robot position
        rp_p = snap_to_finite_resolution(apply_robot_action(s.rp, a))
        # the new time index
        t_idx_p = s.t_idx + 1
        s_p = ProbObstacleSearchState(rp_p, t_idx_p)

        # step cost
        pot = p.prob_obstacle_trees[t_idx_p]
        # Monte Carlo integration over particles
        n_particles = length(pot.data)
        # cost estimate over propagated particle blief
        rm = reward_model(p.model)
        if !isinroom(rp_p, room(p.model))
            # if we left the room, we are done
            continue
        else
            # in any case we will have a negative living penalty
            c = -rm.living_penalty
            # count the number of particles we collided with
            n_particles_collisions = length(inrange(pot, rp_p, agent_min_distance(p.model)))
            collision_prob = n_particles_collisions / n_particles
            if collision_prob > p.collision_prob_thresh
                # don't allow steps where p.collision_prob_tresh is exceeded
                continue
            else
                c -= collision_prob * rm.collision_penalty
            end
            # counte the number of particles that are close to the robot
            n_particles_close = length(inrange(pot, rp_p, 2*agent_min_distance(p.model)))
            c -= n_particles_close / n_particles * rm.dist_to_human_penalty
        end
        push!(successors, (s_p, a, c))
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
    collision_prob_thresh::Float64 = 1e-2
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

POMDPs.solve(sol::ProbObstacleSolver, m::POMDP) = ProbObstaclePolicy(sol, m)

POMDPs.action(po::ProbObstaclePolicy, b) = first(action_info(po, b))

function POMDPModelTools.action_info(po::ProbObstaclePolicy, b; debug=false)
    # Create propagate particles until `sol.max_search_depth` is reached.
    belief_predictions = []
    @assert po.sol.max_search_depth >= 1
    resize!(belief_predictions, po.sol.max_search_depth)
    # first propagation from root belief
    b0 = initialize_belief(po.sol.belief_propagator, b)
    e0 = external(first(particles(b0)))
    rp0 = robot_pos(e0)
    hp0 = human_pos(e0)

    CPUtic()
    belief_predictions[1] = ParticleCollection(predict(po.sol.belief_propagator, decoupled(b0)))
    # recursive propagation of open loop predictions
    for i in 2:po.sol.max_search_depth
        belief_predictions[i] = ParticleCollection(predict(po.sol.belief_propagator, belief_predictions[i-1]))
    end
    prob_obstacle_trees = [KDTree([first(p) for p in particles(bp)]) for bp in belief_predictions]
    prediction_cpu_time = CPUtoq()

    # setup the probabilistic search problem
    heuristic = (s::ProbObstacleSearchState) -> begin
        min_remaining_steps = remaining_step_estimate(po.pomdp, s.rp)
        h = -min_remaining_steps * reward_model(po.pomdp).living_penalty
        return h
    end

    prob_search_problem = ProbObstacleSearchProblem(prob_obstacle_trees=prob_obstacle_trees,
                                                    start_state=ProbObstacleSearchState(rp0, 0),
                                                    model=po.pomdp,
                                                    collision_prob_thresh=po.sol.collision_prob_thresh,
                                                    max_search_depth=po.sol.max_search_depth)

    # solve the probabilistic obstacle avoidance problem using a-star
    aseq::Vector{HSAction}, sseq::Vector{ProbObstacleSearchState} = try
        weighted_astar_search(prob_search_problem, heuristic, 0.2)
    catch e
        if !(e isa InfeasibleSearchProblemError)
            rethrow(e)
        elseif debug
            @warn("No Solution found. Using default action.")
        end
        # use default action (no fault collision) instead
        ([zero(HSAction)], [prob_search_problem.start_state])
    end

    info = (m=po.pomdp,
            belief_predictions=belief_predictions,
            action_sequence=aseq,
            state_sequence=sseq,
            prediction_cpu_time=prediction_cpu_time)

    return first(aseq), info
end

function get_plan(po::ProbObstaclePolicy, belief)
    a, info = action_info(po, belief)
    planning_steps = []
    for i in 1:length(info.action_sequence)
        planning_step = (bp=info.belief_predictions[i],
                         robot_prediction=info.state_sequence[i].rp)
        push!(planning_steps, planning_step)
    end
    return planning_steps
end

