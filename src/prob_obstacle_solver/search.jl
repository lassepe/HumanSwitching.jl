"""
    SearchProblem

A minimalistic, simple interface for search problems
`S` - the state type
`A` - the action type
"""
abstract type SearchProblem{S, A} end
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
    search_node_type

Returns the SearchNode parametrized according to the SearchProblem
"""
search_node_type(::SearchProblem{S, A}) where {S, A} = SearchNode{S, A}
"""
    state_type

Returns the type of the state the search is run over.
"""
state_type(::SearchProblem{S, A}) where {S, A} = S
"""
    action_type

Returns the type of the action the search is run over.
"""
action_type(::SearchProblem{S, A}) where {S, A} = A

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
    child_search_nodes = search_node_type(p)[]
    # TODO: maybe resize in advance or sizehint!
    for (sp, a, c) in successors(p, end_state(n))
        np = SearchNode(vcat(state_sequence(n), sp),
                              vcat(action_sequence(n), [a]), # TODO: vcating this way is a bit risky, maybe add type assert or use copy and push
                              cost(n) + c)
        push!(child_search_nodes, np)
    end

    return child_search_nodes
end
"""
    root_node

Returns a SearchNode over the start_state of this problem
"""
root_node(p::SearchProblem, root_cost::Float64=0.0) = search_node_type(p)([start_state(p)], [], root_cost)

struct InfeasibleSearchProblemError <: Exception
    msg::String
end

function generic_graph_search(p::SearchProblem, fringe_priority::Function)
    # the closed set, states that we don't need to expand anymore
    closed_set = Set{state_type(p)}()
    # the fringe is a priority queue of SearchNode that are left to be expanded
    fringe = PriorityQueue{search_node_type(p), Float64}()

    n0 = root_node(p)
    enqueue!(fringe, n0, fringe_priority(n0))
    while true
        if isempty(fringe)
            throw(InfeasibleSearchProblemError("Fringe was empty, but no solution found"))
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
