"""
HumanBoltzmannModel
"""

abstract type HumanRewardModel end

struct HumanBoltzmannBState <: HumanBehaviorState
    beta::Float64
end

struct HumanBoltzmannToGoalBState <: HumanBehaviorState
    beta::Float64
    goal::Pos
end

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

function human_transition(hbs::HumanBoltzmannBState, hbm::HumanBoltzmannModel, m::HSModel,
                          p::Pos, rng::AbstractRNG)
    hbs_p = hbs
    if rand(rng) < hbm.epsilon
        hbs_p = rand_hbs(rng, hbm)
    end

    # compute the new external state of the human
    return free_evolution(hbm, hbs, p, rng), hbs_p
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
    vel_max::Float64 = 0.4
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

function human_transition(hbs::HumanBoltzmannToGoalBState, hbm::HumanMultiGoalBoltzmann, m::HSModel,
                          p::Pos, rng::AbstractRNG)
    human_pos_p = free_evolution(hbm, hbs, p, rng)
    # with a small likelyhood, resample a new beta
    beta_p = rand(rng) < hbm.beta_resample_sigma ? rand_beta(rng, hbm) : hbs.beta
    # if close to goal, sample next goal according from generative model
    # representing P(g_{k+1} | g_{k})
    goal_p = ((rand(rng) < hbm.goal_resample_sigma || dist_to_pos(human_pos_p, hbs.goal) < agent_min_distance(m)) ?
              hbm.next_goal_generator(hbs.goal, hbm.goals, rng) : hbs.goal)

    hbs_p = HumanBoltzmannToGoalBState(beta_p, goal_p)
    return human_pos_p, hbs_p
end
