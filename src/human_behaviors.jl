"""
# state representations of the human internal state
"""
struct HumanConstVelBState <: HumanBehaviorState
    vx::Float64
    vy::Float64
end

free_evolution(hbs::HumanConstVelBState, p::Pos) = Pos(p.x + hbs.vx, p.y + hbs.vy)

@with_kw struct HumanPIDBState <: HumanBehaviorState
    target_index::Int = 1
    vel_max::Float64 = 0.5
end

target_index(hbs::HumanPIDBState) = hbs.target_index

struct HumanBoltzmannBState <: HumanBehaviorState
    beta::Float64
end

struct HumanBoltzmannToGoalBState <: HumanBehaviorState
    beta::Float64
    goal::Pos
end

"""
# HumanBehaviorModel

Each describe
- from which distribution HumanBehaviorState's are sampled
- how HumanBehaviorState's evolve (see `human_transition_models.jl`)
"""
# basic models don't have further submodels
select_submodel(hbm::HumanBehaviorModel, hbs_type::Type{<:HumanBehaviorState}) = hbm
select_submodel(hbm::HumanBehaviorModel, hbs::HumanBehaviorState)::HumanBehaviorModel = select_submodel(hbm, typeof(hbs))

"""
HumanConstVelBehavior
"""
@with_kw struct HumanConstVelBehavior <: HumanBehaviorModel
    vel_max::Float64 = 1.0
    vel_resample_sigma::Float64 = 0.0
end

bstate_type(::HumanConstVelBehavior)::Type = HumanConstVelBState

# this model randomely generates HumanConstVelBState from the min_max_vel range
rand_hbs(rng::AbstractRNG, hbm::HumanConstVelBehavior) = HumanConstVelBState(rand(rng, Uniform(-hbm.vel_max, hbm.vel_max)),
                                                                             rand(rng, Uniform(-hbm.vel_max, hbm.vel_max)))

@with_kw struct HumanPIDBehavior <: HumanBehaviorModel
    target_sequence::Array{Pos, 1}
end

"""
HumanPIDBehavior
"""
HumanPIDBehavior(r::RoomRep) = HumanPIDBehavior(target_sequence=corner_positions(r))

bstate_type(::HumanBehaviorModel)::Type = HumanPIDBState

rand_hbs(rng::AbstractRNG, hbm::HumanPIDBehavior) = HumanPIDBState(target_index=1)

human_target(hbm::HumanPIDBehavior, hbs::HumanPIDBState) = hbm.target_sequence[target_index(hbs)]
next_target_index(hbm::HumanPIDBehavior, hbs::HumanPIDBState) = min(length(hbm.target_sequence), target_index(hbs)+1)

function free_evolution(hbm::HumanPIDBehavior, hbs::HumanPIDBState, p::Pos)
    human_velocity = min(hbs.vel_max, dist_to_pos(p, human_target(hbm, hbs))) #m/s
    vec2target = vec_from_to(p, human_target(hbm, hbs))
    walk_direction = normalize(vec2target)
    # new position:
    human_pos_p::Pos = p
    if !any(isnan(i) for i in walk_direction)
        xy_p = p[1:2] + walk_direction * human_velocity
        human_pos_p = xy_p
    end

    return human_pos_p
end


"""
HumanBoltzmannModel
"""
abstract type HumanRewardModel end

struct HumanBoltzmannModel{RMT, NA, TA} <: HumanBehaviorModel
    beta_min::Float64
    beta_max::Float64
    betas::Array{Float64}
    epsilon::Float64
    reward_model::RMT

    aspace::SVector{NA, TA}
    _aprob_mem::MVector{NA, Float64}
end

# constructing the boltzmann model in it's most general form
function HumanBoltzmannModel(;beta_min=0.0, beta_max=15.0, betas=[0.0, 15.0],
                              epsilon=0.0, reward_model=HumanSingleTargetRewardModel(),
                              aspace=gen_human_aspace())
    if beta_min == beta_max
        @assert iszero(epsilon)
    end
    return HumanBoltzmannModel(beta_min, beta_max, betas, epsilon, reward_model, aspace,
                              @MVector(zeros(length(aspace))))
end

bstate_type(::HumanBoltzmannModel)::Type = HumanBoltzmannBState

function rand_hbs(rng::AbstractRNG, hbm::HumanBoltzmannModel)
    if length(hbm.betas) > 2
        return HumanBoltzmannBState(rand(rng, hbm.betas))
    end

    return HumanBoltzmannBState(hbm.beta_min == hbm.beta_max ?
                                hbm.beta_max : rand(rng, Truncated(Exponential(5),hbm.beta_min, hbm.beta_max)))
end

@with_kw struct HumanSingleTargetRewardModel
    human_target::Pos = Pos(5, 5)
end

@with_kw struct HumanBoltzmannAction <: FieldVector{2, Float64}
    d::Float64 = 0 # distance
    phi::Float64 = 0 # direction
end

function gen_human_aspace(phi_step::Float64=pi/8)
    dist = 0.5
    direction_actions = [i for i in -pi:phi_step:(pi-phi_step)]
    SVector{length(direction_actions)+1, HumanBoltzmannAction}([zero(HumanBoltzmannAction),(HumanBoltzmannAction(dist, direction) for direction in direction_actions)...])
end

apply_human_action(p::Pos, a::HumanBoltzmannAction)::Pos = Pos(p.x + cos(a.phi)*a.d, p.y + sin(a.phi)*a.d)

function free_evolution(hbm::HumanBoltzmannModel, hbs::HumanBoltzmannBState, p::Pos, rng::AbstractRNG)
    d = get_action_distribution(hbm, hbs, p)
    sampled_action = hbm.aspace[rand(rng, d)]
    p_p = apply_human_action(p, sampled_action)
end

function compute_qval(p::Pos, a::HumanBoltzmannAction, reward_model::HumanSingleTargetRewardModel)
    return -dist_to_pos(apply_human_action(p, a), reward_model.human_target; p=2)
end

function get_action_distribution(hbm::HumanBoltzmannModel, hbs::HumanBoltzmannBState, p::Pos)
    for (i, a) in enumerate(hbm.aspace)
        hbm._aprob_mem[i] = exp(hbs.beta * compute_qval(p, a, hbm.reward_model))
    end
    return Categorical(Array(normalize!(hbm._aprob_mem, 1)))
end

"""
HumanMultiGoalBoltzmann
"""
@with_kw struct HumanMultiGoalBoltzmann{NA, TA} <: HumanBehaviorModel
    beta_min::Float64
    beta_max::Float64
    goals::Array{Pos, 1} = corner_positions(RoomRep())
    next_goal_generator::Function = uniform_goal_generator
    initial_goal_generator::Function = uniform_goal_generator
    vel_max::Float64 = 0.5
    goal_resample_sigma::Float64 = 0.01
    beta_resample_sigma::Float64 = 0.01

    aspace::SVector{NA, TA} = gen_human_aspace()
    _aprob_mem::MVector{NA, Float64} = @MVector(zeros(length(aspace)))
end

bstate_type(hbm::HumanMultiGoalBoltzmann) = HumanBoltzmannToGoalBState

function rand_beta(rng::AbstractRNG, hbm::HumanMultiGoalBoltzmann)
    hbm.beta_min == hbm.beta_max ?
    hbm.beta_max : rand(rng, Truncated(Exponential(5), hbm.beta_min, hbm.beta_max))
end

function rand_hbs(rng::AbstractRNG, hbm::HumanMultiGoalBoltzmann)
    return HumanBoltzmannToGoalBState(rand_beta(rng, hbm),
                                      hbm.initial_goal_generator(hbm.goals, rng))
end

function uniform_goal_generator(goals::Array{Pos, 1}, rng::AbstractRNG)
    return rand(rng, goals)::Pos
end

uniform_goal_generator(::Pos, goals::Array{Pos, 1}, rng::AbstractRNG) = uniform_goal_generator(goals, rng)::Pos

function compute_qval(p::Pos, hbs::HumanBoltzmannToGoalBState, a::HumanBoltzmannAction)
    return -dist_to_pos(apply_human_action(p, a), hbs.goal; p=2)
end

function get_action_distribution(hbm::HumanMultiGoalBoltzmann, hbs::HumanBoltzmannToGoalBState, p::Pos)
    for (i, a) in enumerate(hbm.aspace)
        hbm._aprob_mem[i] = exp(hbs.beta * compute_qval(p, hbs, a))
    end
    return Categorical(Array(normalize!(hbm._aprob_mem, 1)))
end

function free_evolution(hbm::HumanMultiGoalBoltzmann, hbs::HumanBoltzmannToGoalBState, p::Pos, rng::AbstractRNG)
    d = get_action_distribution(hbm, hbs, p)
    sampled_action = hbm.aspace[rand(rng, d)]
    p_p = apply_human_action(p, sampled_action)
end

"""
HumanUniformModelMix
"""
struct HumanUniformModelMix{T} <: HumanBehaviorModel
    submodels::Array{T, 1}
    bstate_change_likelihood::Float64
    bstate_type::Type
end

function HumanUniformModelMix(models...; bstate_change_likelihood::Float64)
    submodels = [models...]
    return HumanUniformModelMix{Union{typeof.(models)...}}(submodels,
                                                           bstate_change_likelihood,
                                                           Union{Iterators.flatten([[bstate_type(sm)] for sm in submodels])...})
end

bstate_type(hbm::HumanUniformModelMix) = hbm.bstate_types

function select_submodel(hbm::HumanUniformModelMix{T}, hbs_type::Type{<:HumanBehaviorState})::T where T
    candidate_submodels = filter(x->(hbs_type <: bstate_type(x)), hbm.submodels)
    @assert(length(candidate_submodels) == 1)
    return first(candidate_submodels)
end

rand_hbs(rng::AbstractRNG, hbm::HumanUniformModelMix)::HumanBehaviorState = rand_hbs(rng::AbstractRNG, rand(rng, hbm.submodels))
