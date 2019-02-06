# the physical representation of a room
@with_kw struct RoomRep
  width::Float64 = 15
  height::Float64 = 15
end

corner_poses(r::RoomRep) = [Pose(x, y, 0) for x in [0.1r.width, 0.9r.width], y in [0.1r.height, 0.9r.height]]

# the physical representation of an agent
@with_kw struct Pose <: FieldVector{3, Float64}
  x::Float64 = 0   # horizontal position
  y::Float64 = 0   # vertical position
  phi::Float64 = 0 # the orientation of the human
end

# defining some sensor/observation models
abstract type HSSensor end
"""
A sensor that gives the exact position (simply extracting the corresponding
portion from the state representation)
"""
struct ExactPositionSensor <: HSSensor end
"""
A sensor that gives a noisy reading to the agents positions
"""
@with_kw struct NoisyPositionSensor <: HSSensor
  # the diagonal of the measurement covariance matrix
  measurement_cov::Array{Float64, 1} = [0.1,
                                        0.1,
                                        0.01]
end
POMDPs.obstype(::ExactPositionSensor) = Pose
POMDPs.obstype(::NoisyPositionSensor) = Pose

# defining some transition models
abstract type HSTransitionModel end
"""
PIDHumanTransition

a deterministic human transition model where the human follows a simiple
proportional controller towards a fixed target
"""
struct PControlledHumanTransition <: HSTransitionModel end
@with_kw struct PControlledHumanAWGNTransition <: HSTransitionModel
  " the diagonal of the transition AWGN covariance matrix"
  pose_cov::Array{Float64, 1} = [0.1, 0.1, 0.01]
end

"""
The state representation of the whole HumanSwitching POMDP

NOTE:
- For now this features only the agent state of a the human (composition and
orientation) as well as its target.
- For now the human is assumed to walk in a straight line towards the unobserved goal
"""
@with_kw struct HSState
  human_pose::Pose
  human_target::Pose
  robot_pose::Pose
  robot_target::Pose
end

function Base.isequal(a::HSState, b::HSState)
  isequal(a.human_pose, b.human_pose) && isequal(a.human_target, b.human_target) &&
  isequal(a.robot_pose, b.robot_pose) && isequal(a.robot_target, b.robot_target)
end

"""
The action representation in the HumanSwitching POMDP

NOTE:
- for now this is not too interesting as the agent can not take any actions
"""
@with_kw struct HSAction
  some_action::Int64 = 0
end

"""
The action space for this problem
"""
struct HSActionSpace end

"""
The MDP representation of the fully observable HumanSwitching problem. This MDP
formulation is used to derive the POMDP formulation for the partially
observable problem.
"""
@with_kw struct HSMDP{TT} <: MDP{HSState, HSAction}
  room::RoomRep = RoomRep()
  transition_model::TT = PControlledHumanTransition()
end

"""
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
  HSPOMDP{typeof(sensor), obstype(sensor), typeof(mdp)}(sensor, mdp)
end

const HSModel = Union{HSMDP, HSPOMDP}
POMDPs.discount(HSModel) = 0.99  # TODO: maybe move to struct field

mdp(m::HSMDP) = m
mdp(m::HSPOMDP) = m.mdp
transition_model(m::HSModel) = mdp(m).transition_model
room(m::HSModel) = mdp(m).room

# Implementing the main API of the generative interface
"""
generate_s

Generates the next state given the last state and the taken action
"""
# this simple forwards to the different transition models
POMDPs.generate_s(m::HSModel, s::HSState, a::HSAction, rng::AbstractRNG)::HSState = generate_s(mdp(m), s, a, rng)

# human controlled by simple P-controller
function human_p_transition(s::HSState)::Tuple{Pose, Pose}
  human_velocity = min(0.6, human_dist_to_target(s)) #m/s
  vec2target = human_vec_to_target(s)
  target_direction = normalize(vec2target)
  current_walk_direction = @SVector [cos(s.human_pose.phi), sin(s.human_pose.phi)]
  walk_direction = (target_direction + current_walk_direction)/2
  # new position:
  human_pose_p::Pose = s.human_pose
  if !any(isnan(i) for i in target_direction)
    xy_p = s.human_pose[1:2] + walk_direction * human_velocity
    phi_p = atan(walk_direction[2], walk_direction[1])
    human_pose::Pose = [xy_p..., phi_p]
  end

  return human_pose, s.human_target
end

# helper funciton to access the deterministic P controlled human transition
function POMDPs.generate_s(m::HSMDP{PControlledHumanTransition}, s::HSState, a::HSAction, rng::AbstractRNG)::HSState
  # assembling the new state
  human_pose_p, human_target_p = human_p_transition(s)
  # TODO: robotAction
  robot_pose_p = Pose(0, 0, 0)
  robot_target_p = Pose(0, 0, 0)

  HSState(human_pose_p, human_target_p, robot_pose_p, robot_target_p)
end

# same as above but with AWGN
function POMDPs.generate_s(m::HSMDP{PControlledHumanAWGNTransition}, s::HSState, a::HSAction, rng::AbstractRNG)::HSState
  # first get the deterministic version
  human_pose_p, human_target_p = human_p_transition(s)
  # add AWGN to the pose
  # TODO: Maybe we also want noise on the target
  do_resample = rand(rng) > 0.99
  human_target_p = do_resample ? rand(corner_poses(room(m))) : human_target_p

  HSState(human_pose_p + rand(rng, MvNormal([0, 0, 0], transition_model(m).pose_cov)), human_target_p,
          Pose(), Pose()) # TODO: robotAction
end

# TODO: robotObservation
@with_kw struct HSObservation
  human_pose::Pose
  robot_pose::Pose
  robot_target::Pose
end
# some convenient constructor
HSObservation(s::HSState) = HSObservation(s.human_pose, s.robot_pose, s.robot_target)

"""
generate_o

Generates an observation for an observed transition
"""
# TODO: This is a bit misleading as it make it look like the full state was
# observable. `Pose` should probably be renamed.

# In this version the observation is a **deterministic** extraction of the observable part of the state
# TODO: robotObservation
POMDPs.generate_o(m::HSPOMDP{ExactPositionSensor, Pose}, s::HSState, a::HSAction, sp::HSState, rng::AbstractRNG)::Pose = sp.human_pose

# In this version the observation is a **noisy** extraction of the observable part of the state
function POMDPs.generate_o(m::HSPOMDP{NoisyPositionSensor, Pose}, s::HSState, a::HSAction, sp::HSState, rng::AbstractRNG)::Pose
  # NOTE: this distribution is **already** centered around the state sp
  o_distribution = MvNormal(convert(Array, sp.human_pose), m.sensor.measurement_cov)
  return rand(rng, o_distribution)
end

# TODO: Think about this
#
# This needs a lot of boiler plate code since we also need to know the pdf on
# this distribution etc.  Implementing a custom update might make more sense
# (and is neccessary for other reasons anyway)
#
function POMDPs.observation(m::HSModel, s::HSState)
  return MvNormal(convert(Array, s.human_pose), m.sensor.measurement_cov)
end

"""
isterminal

checks if the state is a terminal state

TODO: For now the problem terminates if the human reached it's goal. This
should be done differently as soon as the agent is doing more than just
observing the scene
"""

# TODO: robotAction (robot also needs to reach target)
POMDPs.isterminal(m::HSModel, s::HSState) = human_dist_to_target(s) < 0.1

"""
initialstate

Draw an initial state and a target state for the human agent.
"""
# TODO: Later this will also include the start and goal of the robot agent
function POMDPs.initialstate(m::HSModel, rng::AbstractRNG)::HSState
  # generate an initial position and a goal for the human
  human_init_pose = rand_pose(room(m), rng=rng)
  # for now the target is one of the 4 corners of the room
  human_target_pose = rand(rng, corner_poses(room(m)))

  # the robot starts in some rando mpose and want's to go some random pose
  robot_init_pose = rand_pose(room(m), rng=rng)
  robot_target_pose = rand_pose(room(m), rng=rng)

  return HSState(human_pose=human_init_pose, human_target=human_target_pose,
                 robot_pose=robot_init_pose, robot_target=robot_target_pose)
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
function POMDPs.reward(m::HSModel, s::HSState, a::HSAction)::Float64
  return 0.0
end
