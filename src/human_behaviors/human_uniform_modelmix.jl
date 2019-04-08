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

function human_transition(hbs::HumanBehaviorState, hbm::HumanUniformModelMix, m::HSModel,
                          p::Pos, rng::AbstractRNG)
    # select the corresponding sub model
    hbm_sub = select_submodel(hbm, hbs)
    # propagate state according to this model
    human_pos_p, hbs_sub_prime = human_transition(hbs, hbm_sub, m, p, rng)
    # small likelihood of randomly changing behavior state
    hbs_p = rand(rng) < hbm.bstate_change_likelihood ? rand_hbs(rng, hbm) : hbs

    return human_pos_p, hbs_p
end
