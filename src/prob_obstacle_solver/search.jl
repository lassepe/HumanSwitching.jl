"""
    SearchProblem

A minimalistic, simple interface for search problems
`S` is the state type
"""
abstract type SearchProblem{S} end
"""
    start_state(p::SearchProblem)

Returns the start state of the problem.
"""
function start_state end
"""
    is_goal_state(p::SearchProblem, s::S)

Returns true if a given state is a goal state
"""
function is_goal_state end
"""
    successors(p::SearchProblem, s::S)

Returns a list of tuples `(sp, a, c)`, i.e. the next state, the action that
yields this state and the cost for this transition.
"""
function successors end

"""
    SearchNode

Describes a path along the graph. Containting states, action and cost.
"""
struct SearchNode{S, A}
    "The sequence of states on the path this node represents."
    state_sequence::Vector{S}
    "The sequence of actions on neccesary to traverse the state trajectory represented by this node."
    action_sequence::Vector{A}
    "The cumulative cost for traversing the state trajectory represented by this node."
    cost::Float64
end
"""
    cost(n::SearchNode)

Returns the cost for traversing the path described by the node.
"""
cost(n::SearchNode) = n.cost
"""
    end_state(n::SearchNode)

Return the last state of the path described by the node.
"""
end_state(n::SearchNode) = last(n.state_sequence)
"""
    action_sequence(n::SearchNode)

Returns the action sequence neccessary to traverse the path to the `end_state`
"""
action_sequence(n::SearchNode) = n.action_sequence
"""
    state_sequence(n::SearchNode)

Returns the sequence of states along the path represented by this node
"""
state_sequence(n::SearchNode) = n.state_sequence
"""
    expand(n::SearchNode)

Returns the child search nodes (set of next, longer pathes from this node)
"""
function expand(n::SearchNode, p::SearchProblem)
    child_search_nodes = []
    # TODO: maybe resize in advance or sizehint!
    for (sp, a, c) in successors(p, end_state(n))
        push!(child_search_nodes,
              SearchNode(vcat(state_sequence(n), sp),
                         vcat(action_sequence(n), [a]), # TODO: vcating this way is a bit risky, maybe add type assert or use copy and push
                         cost(n) + c))
    end

    return child_search_nodes
end



function generic_graph_search(p::SearchProblem{S}, fringe_priority::Function) where S
    # the closed set, states that we don't need to expand anymore
    closed_set = Set{S}()
    # the fringe is a priority queue of SearchNode that are left to be expanded
    fringe = PriorityQueue()

    n0 = SearchNode([start_state(p)], [], 0.0)
    enqueue!(fringe, n0, fringe_priority(n0))
    while true
        if isempty(fringe)
            throw(ErrorException("Fringe was empty, but no solution found"))
        end
        current_search_node = dequeue!(fringe)
        # We have found a path to the goal state
        if is_goal_state(p, end_state(current_search_node))
            return action_sequence(current_search_node), state_sequence(current_search_node)
        elseif !(end_state(current_search_node) in closed_set)
            # make sure we don't explore this state again
            push!(closed_set, end_state(current_search_node))
            # expand the node
            for child_search_node in expand(current_search_node, p)
                enqueue!(fringe, child_search_node, fringe_priority(child_search_node))
            end
        else
            continue
        end
    end
end

astar_search(p::SearchProblem, heuristic::Function) = weighted_astar_search(p, heuristic, 0.0)

function weighted_astar_search(p::SearchProblem, heuristic::Function, eps::Float64)
    @assert eps >= 0.0
    astar_priority = (n::SearchNode) -> cost(n) + (1+eps)*heuristic(end_state(n))
    return generic_graph_search(p, astar_priority)
end
