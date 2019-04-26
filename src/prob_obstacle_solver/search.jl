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
    "The  state at the end of the path represented by this node"
    leaf_state::S
    "The  action at the end of the path represented by this node"
    leaf_action::Union{A, Nothing}
    "The cumulative cost for traversing the state trajectory represented by this node."
    cost::Float64
    "The parent node, if not a root node"
    parent::Union{SearchNode{S, A}, Nothing}
    "The depth of this node in the tree."
    depth::Int
end
"""
    state_type

Returns the type of the state the search is run over.
"""
state_type(::SearchNode{S, A}) where {S, A} = S
"""
    action_type

Returns the type of the action the search is run over.
"""
action_type(::SearchNode{S, A}) where {S, A} = A
"""
    depth(n::SearchNode)

The depth of the search node in the tree.
"""
depth(n::SearchNode) = n.depth
"""
    parent(n::SearchNode)

The parent node of the given search node. `nothing` if n is a root node.
"""
parent(n::SearchNode) = n.parent
"""
    cost(n::SearchNode)

Returns the cost for traversing the path described by the node.
"""
cost(n::SearchNode) = n.cost
"""
    leaf_state(n::SearchNode)

Returns the last state of the path described by the node.
"""
leaf_state(n::SearchNode) = n.leaf_state
"""
    leaf_action(n::SearchNode)

Returns the last action on the path described by this node.
"""
leaf_action(n::SearchNode) = n.leaf_action
"""
    action_sequence(n::SearchNode)

Returns the action sequence neccessary to traverse the path to the `leaf_state`
"""
function action_sequence(n::SearchNode)
    action_sequence = action_type(n)[]
    resize!(action_sequence, depth(n))

    current_node = n
    # traversing the path of the node in reverse order
    for i in reverse(1:depth(n))
        action_sequence[i] = leaf_action(current_node)
        current_node = parent(current_node)
    end
    # at the end of this path there must be a root node.
    # The parent of a root node is nothing!
    @assert isnothing(parent(current_node))

    return action_sequence::Vector{action_type(n)}
end
"""
    state_sequence(n::SearchNode)

Returns the sequence of states along the path represented by this node
"""
function state_sequence(n::SearchNode)
    state_sequence = state_type(n)[]
    resize!(state_sequence, depth(n))

    current_node = n
    # traversing the path of the node in reverse order
    for i in reverse(1:depth(n))
        state_sequence[i] = leaf_state(current_node)
        current_node = parent(current_node)
    end
    # at the end of this path there must be a root node.
    # The parent of a root node is nothing!
    @assert isnothing(parent(current_node))

    return state_sequence::Vector{state_type(n)}
end
"""
    expand(n::SearchNode)

Returns the child search nodes (set of next, longer pathes from this node)
"""
function expand(n::SearchNode, p::SearchProblem)
    # TODO: maybe resize in advance or sizehint!
    child_search_nodes = search_node_type(p)[]
    for (sp, a, c) in successors(p, leaf_state(n))
        np = SearchNode(sp, a, cost(n) + c, n, depth(n)+1)
        push!(child_search_nodes, np)
    end

    return child_search_nodes
end
"""
    root_node

Returns a SearchNode over the start_state of this problem
"""
root_node(p::SearchProblem, root_cost::Float64=0.0) = search_node_type(p)(start_state(p),
                                                                          nothing,
                                                                          root_cost,
                                                                          nothing,
                                                                          0)

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
        if is_goal_state(p, leaf_state(current_search_node))
            return action_sequence(current_search_node), state_sequence(current_search_node)
        elseif !(leaf_state(current_search_node) in closed_set)
            # make sure we don't explore this state again
            push!(closed_set, leaf_state(current_search_node))
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
    astar_priority = (n::SearchNode) -> cost(n) + (1+eps)*heuristic(leaf_state(n))
    return generic_graph_search(p, astar_priority)
end
