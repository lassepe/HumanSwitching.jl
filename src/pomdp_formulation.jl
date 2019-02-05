# the physical representation of a room
@with_kw struct RoomRep
  width::Float64 = 15
  height::Float64 = 15
end

corner_states(r::RoomRep) = [AgentState(xy=[x, y], phi=0) for x in [0.1r.width, 0.9r.width], y in [0.1r.height, 0.9r.height]]

# the physical representation of an agent
@with_kw struct AgentState
  xy::Array{Float64, 1} = [0, 0]# the x- and y-position
  phi::Float64 = 0 # the orientation of the human
end

function Base.isequal(a::AgentState, b::AgentState)
  return isequal(a.xy, b.xy) && isequal(a.phi, b.phi)
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
  # the diagonal of the measurement covariance matrix
  measurement_cov::Array{Float64, 1} = [0.1,
                                        0.1,
                                        0.01]
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

function Base.isequal(a::HSState, b::HSState)
  isequal(a.human_pose, b.human_pose) && isequal(a.human_target, b.human_target)
end

"""
The action representation in the HumanSwitching POMDP

NOTE:
- for now this is not too interesting as the agent can not take any actions
"""
struct HSAction
  some_action::Int64
end
HSAction() = HSAction(0)

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
@with_kw struct HSMDP{AS} <: MDP{HSState, HSAction}
  room::RoomRep = RoomRep()
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
@with_kw struct HSPOMDP{TS, O} <: POMDP{HSState, HSAction, O}
  sensor::TS = ExactPositionSensor()
  mdp::HSMDP = HSMDP()
end
# TODO: maybe internalize


POMDPs.discount(HSModel) = 0.99
const HSModel = Union{HSMDP, HSPOMDP}

# Some convenient constructors
HSPOMDP(sensor::Union{ExactPositionSensor, NoisyPositionSensor}) = HSPOMDP{typeof(sensor), obstype(sensor)}(sensor, HSMDP())
HSPOMDP(sensor::Union{ExactPositionSensor, NoisyPositionSensor}, mdp::HSMDP) = HSPOMDP{typeof(sensor), obstype(sensor)}(sensor, mdp)

mdp(m::HSMDP) = m
mdp(m::HSPOMDP) = m.mdp
room(m::HSModel) = mdp(m).room

POMDPs.actions(m::HSModel) = mdp(m).aspace
POMDPs.n_actions(m::HSModel) = length(mdp(m).aspace)

# Implementing the main API of the generative interface
"""
generate_s

Generates the next state given the last state and the taken action
"""
function POMDPs.generate_s(p::HSModel, s::HSState, a::HSAction, rng::AbstractRNG)::HSState
  # TODO: For now the human simply has a P controller that drives him to the
  # goal with a constant velocity. On a long run one can implement different
  # models for the human (e.g. Bolzmann)
  human_velocity = min(0.6, human_dist_to_target(s)) #m/s
  vec2target = human_vec_to_target(s)
  target_direction = normalize(vec2target)
  current_walk_direction = @SVector [cos(s.human_pose.phi), sin(s.human_pose.phi)]
  walk_direction = (target_direction + current_walk_direction)/2
  # new position:
  if any(isnan(i) for i in target_direction)
    xy_p = s.human_pose.xy
    phi_p = s.human_pose.phi
  else
    xy_p = s.human_pose.xy +  walk_direction * human_velocity
    phi_p = atan(walk_direction[2], walk_direction[1])
  end
  return HSState(human_pose=AgentState(xy=xy_p, phi=phi_p), human_target=s.human_target)
end
"""
generate_o

Generates an observation for an observed transition
"""
# TODO: This is a bit misleading as it make it look like the full state was
# observable. `AgentState` should probably be renamed.

# In this version the observation is a **deterministic** extraction of the observable part of the state
function POMDPs.generate_o(p::HSPOMDP{ExactPositionSensor, AgentState}, s::HSState, a::HSAction, sp::HSState, rng::AbstractRNG)::AgentState
  return sp.human_pose
end

# In this version the observation is a **noisy** extraction of the observable part of the state
function POMDPs.generate_o(p::HSPOMDP{NoisyPositionSensor, AgentState}, s::HSState, a::HSAction, sp::HSState, rng::AbstractRNG)::AgentState
  pose_vec = [sp.human_pose.xy..., sp.human_pose.phi]
  # NOTE: this distribution is **already** centered around the state sp
  o_distribution = MvNormal(pose_vec, p.sensor.measurement_cov)
  o_vec = rand(rng, o_distribution)
  return AgentState(xy=o_vec[1:2], phi=o_vec[3])
end

# TODO: Think about this
#
# This needs a lot of boiler plate code since we also need to know the pdf on
# this distribution etc.  Implementing a custom update might make more sense
# (and is neccessary for other reasons anyway)
#
# struct HSObservationDistribution
#   m::HSModel
#   s::HSState
# end
#
# POMDPs.observation(m::HSModel, s::HSState) = HSObservationDistribution(m, s)
# POMDPs.rand(rng::AbstractRNG, d::HSObservationDistribution) = generate_o(d.m, d.s, HSAction(), HSState(), rng)

"""
isterminal

checks if the state is a terminal state

TODO: For now the problem terminates if the human reached it's goal. This
should be done differently as soon as the agent is doing more than just
observing the scene
"""
POMDPs.isterminal(m::HSModel, s::HSState) = human_dist_to_target(s) < 0.1

"""
initialstate

Draw an initial state and a target state for the human agent.
"""
# TODO: Later this will also include the start and goal of the robot agent
function POMDPs.initialstate(p::HSModel, rng::AbstractRNG)::HSState
  # generate an initial position and a goal for the human
  human_init_state = rand_astate(room(p), rng=rng)
  # for now the target is one of the 4 corners of the room
  human_target_state = rand(rng, corner_states(room(p)))
  return HSState(human_pose=human_init_state, human_target=human_target_state)
end

struct HSInitialDistribution{ModelType<:HSModel}
  m::ModelType
end

"""
rand

Defines how to sample from a HSInitialDistribution
"""
POMDPs.rand(rng::AbstractRNG, d::HSInitialDistribution) = initialstate(d.m, rng)

"""
initialstate_distribution

defines the initial state distribution on this
problem set which can be passed to rand """
POMDPs.initialstate_distribution(m::HSModel) = HSInitialDistribution(m)

"""
reward

The reward function for this problem.

NOTE: Nothing intereseting here until the agent is also moving
"""
function POMDPs.reward(p::HSModel, s::HSState, a::HSAction)::Float64
  return 0.0
end
