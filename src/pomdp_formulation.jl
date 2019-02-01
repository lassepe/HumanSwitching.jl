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

"""
A sensor that gives the exact position (simply extracting the corresponding
portion from the state representation)
"""

struct ExactPositionSensor end
"""
A sensor that gives a noisy reading to the agents positions
"""
@with_kw struct NoisyPositionSensor
  measurement_cov::SMatrix{3,3} = NoisyPositionSensor([1,0,0,
                                                       0,1,0,
                                                       0,0,0.1])
end

# Some convenient aliases
const HSSensor = Union{ExactPositionSensor,
                       NoisyPositionSensor}

POMDPs.obstype(::ExactPositionSensor) = AgentState
POMDPs.obstype(::NoisyPositionSensor) = AgentState

"""
The state representation of the whole HumanSwitching POMDP

NOTE:
- For now this features only the agent state of a the human (composition and
orientation) as well as its target.
- For now the human is assumed to walk in a straight line towards the unobserved goal
"""
@with_kw struct HSState
  human_pose::AgentState
  human_target::AgentState
end

"""
The action representation in the HumanSwitching POMDP

NOTE:
- for now this is not too interesting as the agent can not take any actions
"""
struct HSAct
  some_action::Int64
end
HSAct() = HSAct(0)

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
@with_kw struct HSMDP{AS} <: MDP{HSState, HSAct}
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
- `sensor::TS` a struct specifying the sensor used (e.g. ExactPositionSensor)
- `mdp::HSDMP` a MDP version of this problem
"""
@with_kw struct HSPOMDP{TS, O} <: POMDP{HSState, HSAct, O}
  sensor::TS = ExactPositionSensor()
  mdp::HSMDP = HSMDP()
end
const HSModel = Union{HSMDP, HSPOMDP}

# Some convenient constructors
HSPOMDP(sensor::Union{ExactPositionSensor, NoisyPositionSensor}, mdp::HSMDP) = HSPOMDP{typeof(sensor), obstype(sensor)}(sensor, mdp)

mdp(m::HSMDP) = m
mdp(m::HSPOMDP) = m.mdp
POMDPs.actions(m::HSModel) = mdp(m).aspace
POMDPs.n_actions(m::HSModel) = length(mdp(m).aspace)

# Implementing the main API of the generative interface
"""
generate_s

Generates the next state given the last state and the taken action
"""

function generate_s(p::HSModel, s::HSState, a::HSAct, rng::AbstractRNG)::HSState
  # TODO: For now the human simply has a P controller that drives him to the
  # goal with a constant velocity. On a long run one can implement different
  # models for the human (e.g. Bolzmann)
  human_velocity = 1 #m/s
  vec2target = human_vec_to_target(s)
  walking_direction = normalize(vec2target)
  # new position:
  if any(isnan(i) for i in walking_direction)
    xy_p = s.human_pose.xy
    phi_p = s.human_pose.phi
  else
    xy_p = s.human_pose.xy + walking_direction * human_velocity
    phi_p = human_angle_to_target(s)
  end
  return HSState(human_pose=AgentState(xy=xy_p, phi=phi_p), human_target=s.human_target)
end
