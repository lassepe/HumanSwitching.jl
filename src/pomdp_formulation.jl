# the physical representation of a room
@with_kw struct RoomRep
  width::Float64 = 15
  height::Float64 = 15
end

# the physical representation of an agent
@with_kw struct Pose <: FieldVector{3, Float64}
  x::Float64 = 0   # horizontal position
  y::Float64 = 0   # vertical position
  phi::Float64 = 0 # the orientation of the human
end

"""
HSSensor

Abstract type for different sensor models
"""
abstract type HSSensor end
abstract type HSPostTransitionTransform end

struct ExactPositionSensor <: HSSensor end

@with_kw struct NoisyPositionSensor <: HSSensor
  # the diagonal of the measurement covariance matrix
  measurement_cov::Array{Float64, 1} = [0.1,
                                        0.1,
                                        0.01]
end

# TODO: refactorState
# - this might also need to include other model details
abstract type HumanBehaviorModel end

@with_kw struct HSExternalState
  human_pose::Pose
  robot_pose::Pose
  robot_target::Pose
end

@with_kw struct HSState
  external::HSExternalState
  hbm::HumanBehaviorModel
end

function HSState(human_pose::Pose, human_target::Pose,
                 robot_pose::Pose, robot_target::Pose)

  HSState(external=HSExternalState(human_pose,
                                   robot_pose,
                                   robot_target),
          # TODO: humanBehavior - for now this is fixed but this should be
          # selected dynamically from a list
          hbm=HumanPIDBehavior(human_target=human_target))
end

external(s::HSState) = s.external
external(s::HSExternalState) = s
hbm(s::HSState) = s.hbm
hbm(m::HumanBehaviorModel) = m

human_target(s::Union{HSState, HumanBehaviorModel}) = hbm(s).human_target
human_pose(s::Union{HSState, HSExternalState}) = external(s).human_pose
robot_pose(s::Union{HSState, HSExternalState}) = external(s).robot_pose
robot_target(s::Union{HSState, HSExternalState}) = external(s).robot_target

@with_kw struct HSObservation <: FieldVector{5, Float64}
  h_x::Float64 = 0
  h_y::Float64 = 0
  h_phi::Float64 = 0
  r_x::Float64 = 0
  r_y::Float64 = 0
end
# some convenient constructor
HSObservation(s::HSState) = HSObservation(human_pose(s)..., robot_pose(s)[1:2]...)

@with_kw struct HSAction <: FieldVector{2, Float64}
  d::Float64 = 0   # distance to travel
  phi::Float64 = 0 # angle of direction in which the distance is traveled
end

struct HSActionSpace
  actions::AbstractVector{HSAction}
end

# defining the default action space
function HSActionSpace()
  dist_actions = (0.2, 0.4)
  phi_resolution = (pi/4)
  phi_actions = (-pi:phi_resolution:(pi-phi_resolution))

  return vec([zero(HSAction()), (HSAction(d, phi) for d in dist_actions, phi in phi_actions)...])
end

apply_action(p::Pose, a::HSAction) = Pose(p.x + cos(a.phi)*a.d, p.y + sin(a.phi)*a.d, p.phi)

"""
HSMDP

The MDP representation of the fully observable HumanSwitching problem. This MDP
formulation is used to derive the POMDP formulation for the partially
observable problem.

Parameters:

- `AS` the type of the action space
"""
@with_kw struct HSMDP{AS} <: MDP{HSState, HSAction}
  room::RoomRep = RoomRep()
  aspace::AS = HSActionSpace()
  reward_model = HSRewardModel()
  agent_min_distance::Float64 = 1.0
  post_transition_transform::HSPostTransitionTransform = HSIdentityPTT()
  known_external_initstate::Union{HSExternalState, Nothing} = nothing
end

"""
HDPOMDP

The POMDP representation (thus partially observable) formulation of the
HumanSwitching problem.

Fields:

- `sensor::TS` a struct specifying the sensor used
- `mdp::HSDMP` a MDP version of this problem

Parameters:

- `TS` the type of the sensor
- `O` the type of of observations the sensor model returns
- `M` the type of the underlying MDP
"""
@with_kw struct HSPOMDP{TS, O, M} <: POMDP{HSState, HSAction, O}
  sensor::TS = ExactPositionSensor()
  mdp::M = HSMDP()
end

function HSPOMDP(sensor::HSSensor, mdp::HSMDP)
  HSPOMDP{typeof(sensor), HSObservation, typeof(mdp)}(sensor, mdp)
end

const HSModel = Union{HSMDP, HSPOMDP}

mdp(m::HSMDP) = m
mdp(m::HSPOMDP) = m.mdp
reward_model(m::HSModel) = mdp(m).reward_model
post_transition_transform(m::HSModel) = mdp(m).post_transition_transform

room(m::HSModel) = mdp(m).room
agent_min_distance(m::HSModel) = mdp(m).agent_min_distance

POMDPs.actions(m::HSModel) = mdp(m).aspace
POMDPs.n_actions(m::HSModel) = length(mdp(m).aspace)
POMDPs.discount(m::HSModel) = reward_model(m).discount_factor

"""
generate_s

Generates the next state given the last state and the taken action
"""
# this simple forwards to the different transition models
function POMDPs.generate_s(m::HSModel, s::HSState, a::HSAction, rng::AbstractRNG)::HSState
  @assert (a in actions(m))

  # compute the human transition - giving a new pose for the human and a new transition model
  mp, human_pose_p = human_transition(hbm(s), human_pose(s))

  # compute the transition of the robot
  robot_pose_p = apply_action(robot_pose(s), a)
  robot_target_p = robot_target(s)

  sp::HSState = HSState(external=HSExternalState(human_pose_p,
                                                  robot_pose_p,
                                                  robot_target_p),
                         hbm=mp)

  # potentially add some noise to sp for numerical reasons
  return post_transition_transform(m, s, a, sp,
                                   rng::AbstractRNG)::HSState
end

"""
generate_o

Generates an observation for an observed transition
"""

# In this version the observation is a **deterministic** extraction of the observable part of the state
POMDPs.generate_o(m::HSPOMDP{ExactPositionSensor, HSObservation, <:Any},
                  s::HSState, a::HSAction, sp::HSState, rng::AbstractRNG)::HSObservation = HSObservation(sp)
POMDPs.generate_o(m::HSPOMDP{NoisyPositionSensor, HSObservation, <:Any},
                  s::HSState, a::HSAction, sp::HSState, rng::AbstractRNG)::HSObservation = HSObservation(rand(rng, POMDPs.observation(m, sp)))

# TODO: This is a bit ugly. There should be away to directly define a distribution type on a FieldVector
function POMDPs.observation(m::HSPOMDP{NoisyPositionSensor, HSObservation, <:Any}, s::HSState)
  # TODO: do this properly
  return MvNormal(Array{Float64, 1}([human_pose(s)..., robot_pose(s)[1:2]...]),
                  Array{Float64, 1}([m.sensor.measurement_cov..., m.sensor.measurement_cov[1:2]...]))
end

"""
isterminal

checks if the state is a terminal state
"""
function POMDPs.isterminal(m::HSModel, s::HSState)
  robot_reached_target(s) || !isinroom(robot_pose(s), room(m)) || has_collision(m, s)
end

"""
initialstate

Sample an initial state and a target state for each agent.
"""
function POMDPs.initialstate(m::HSModel, rng::AbstractRNG)::HSState
  return rand_state(room(m), rng; known_external_state=mdp(m).known_external_initstate)
end

struct HSInitialDistribution{ModelType<:HSModel}
  m::ModelType
end

POMDPs.rand(rng::AbstractRNG, d::HSInitialDistribution) = initialstate(d.m, rng)
POMDPs.initialstate_distribution(m::HSModel) = HSInitialDistribution(m)
