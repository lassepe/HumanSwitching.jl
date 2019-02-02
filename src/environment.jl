# the physical representation of a room
@with_kw struct RoomRep
  width::Float64 = 20
  height::Float64 = 20
end

# the physical representation of an agent
@with_kw struct AgentState
  xy::SVector{2, Float64} = [0, 0]# the x- and y-position
  phi::Float64 = 0 # the orientation of the human
end

# TODO: implement differen sensor types
"""
A sensor that gives the exact position (simply extracting the corresponding
portion from the state representation)
"""

struct ExactPositionSensor end
"""
A sensor that gives a noisy reading to the agents positions
"""
@with_kw struct NoisyPositionSensor
  measurement_cov::SMatrix{3,3}
end
NoisyPositionSensor() = NoisyPositionSensor([1,0,0;
                                             0,1,0;
                                             0,0,0.1])

POMDPs.obstype(::ExactPositionSensor) = AgentState
POMDPs.obstype(::NoisyPositionSensor) = AgentState

"""
The state representation of the whole HumanSwitching POMDP

NOTE:
- For now this features only the agent state of a the human (composition and
orientation) as well as its target.
- For now the human is assumed to walk in a straight line towards the unobserved goal
"""
struct HSState
  human_pose::AgentState
  human_target::AgentState
end

"""
The action representation in the HumanSwitching POMDP

NOTE:
- for now this is not too interesting as the agent can not take any actions
"""
struct HSAct
  some_action::Float64
end

"""
The action space for this problem
"""
struct HSActionSpace end

"""
The MDP representation of the fully observable HumanSwitching problem. This MDP
formulation is used to derive the POMDP formulation for the partially
observable problem.

- state transitions / generation
- reward assignments

NOTE: In a more mature version of this code this should also feature:
- physical limitations (e.g. speed of robot)
- terminal check

For now this is rather glue code as we will implement this version is a pure
information gathering problem for now.
"""
@with_kw mutable struct HSMDP{SS,AS} <: MDP{HSState, HSAct}
  aspace::AS = HSActionSpace()
end

"""
The POMDP representation (thus partially observable) formulation of the
HumanSwitching problem.

NOTE:
For now we will assume that the positions observable with absolute certainty.
This has some implications to think of later:

- the observation function of the position is a dirac and thus will cause
trouble in observation update. This state as to externalized.

Fields:
- `sensor::T` a struct specifying the sensor used (e.g. ExactPositionSensor)
- `mdp::HSDMP` a MDP version of this problem

NOTE:
- implement different sensor types
"""
struct HSPOMDP{T, O} <: POMDP{HSState, HSAct, O}
  sensor::T,
  mdp::HSMDP
end
mdp(m::HSMDP) = m
mdp(m::HSMDP) = m.mdp
"""
Some convenient constructors
"""
HSPOMDP(sensor::Union{ExactPositionSensor, NoisyPositionSensor}, mdp::HSMDP) = HSPOMDP{typeof(sensor), obstype(sensor)}(sensor, mdp)

POMDP.actions(m::Union{HSMDP, HSPOMDP}) = mdp(m).aspace
POMDP.n_actions(m::Union{HSMDP, HSPOMDP}) = length(mdp(m).aspace)
