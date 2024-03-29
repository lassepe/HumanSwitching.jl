"""
SharedExternalStateBelief

A belief representation where all particles share some part of the state (the
external state) and are distributed only over a subset of the dimensions, e.g.
the internal state

Parameters:
- `E` the type of the external part of the state. This should be the same type as the observation!
- `I` the type of the internal part of the state
- `S` the type of the full state (internal + external)

"""
mutable struct SharedExternalStateBelief{E, I, S} <: AbstractParticleBelief{S}
    " the fully observable part of the state (positions of all agents and robot goal) "
    external::E
    " particles to keep track of the hidden part of the state "
    internal_particles::Vector{I}
    " weights for each particle"
    weights::Vector{Float64}
    " the sum of all weights "
    weight_sum::Float64
end


function SharedExternalStateBelief{E, I, S}(particles::AbstractVector{S}, weights::AbstractVector{Float64}, weight_sum=sum(weights)) where {E, I, S}
    return SharedExternalStateBelief{E, I, S}(external(first(particles))::E, [internal(p)::I for p in particles], weights, weight_sum)
end

"""
The user is expected to provide methods:

- `compose_state(se::E, si::I)::S`
- `external`(s::S)::E
- `internal`(s::S)::I

the full state of type `S` from objects of type `I` and `S`.
"""

function compose_state end
function external end
function internal end

external_type(st::Type{S}) where {S} = first(Base.return_types(external, (st, )))
internal_type(st::Type{S}) where {S} = first(Base.return_types(internal, (st, )))

# determines the types of the external, internal and full state representation
# from a given belief
function ext_int_typesig(b::B) where {B <: AbstractParticleBelief}
    s_type = sampletype(b)
    return (external_type(s_type), internal_type(s_type), s_type)
end

ParticleFilters.n_particles(b::SharedExternalStateBelief) = length(b.internal_particles)
ParticleFilters.particles(b::SharedExternalStateBelief{E, I, S}) where {E, I, S} = [compose_state(b.external, p)::S for p in b.internal_particles]
ParticleFilters.weighted_particles(b::SharedExternalStateBelief) = (compose_state(b.external, b.internal_particles[i])=>b.weights[i]
                                                                    for i in 0:length(b.internal_particles))
ParticleFilters.weight_sum(b::SharedExternalStateBelief) = b.weight_sum
ParticleFilters.weight(b::SharedExternalStateBelief, i::Int) = b.weights[i]
ParticleFilters.particle(b::SharedExternalStateBelief{E, I, S}, i::Int) where {E, I, S} = compose_state(b.external, b.internal_particles[i])::S
ParticleFilters.weights(b::SharedExternalStateBelief) = b.weights

function Random.rand(rng::AbstractRNG, b::SharedExternalStateBelief)
    t = rand(rng) * weight_sum(b)
    i = 1
    cw = weights(b)[1]
    while cw < t && i < length(b.weights)
        i += 1
        @inbounds cw += weights(b)[i]
    end
    return particles(b)[i]
end
Statistics.mean(b::SharedExternalStateBelief) = dot(weights(b), particles(b))/weight_sum(b)

# For now it should be enough to have a domain specific resampler
struct SharedExternalStateResampler
    lv::LowVarianceResampler
end

SharedExternalStateResampler(n::Int) = SharedExternalStateResampler(LowVarianceResampler(n))

function ParticleFilters.resample(rs::SharedExternalStateResampler,
                                  bp::SharedExternalStateBelief,
                                  pm::POMDP,
                                  rm::POMDP,
                                  b,
                                  a,
                                  o,
                                  rng::AbstractRNG)
    # we first update the external component of the
    bp.external = o
    # now we can use low variance resampling to do the rest of the job
    return resample(rs.lv, bp, rng)
end

function ParticleFilters.resample(rs::SharedExternalStateResampler,
                                  bp::WeightedParticleBelief,
                                  pm::POMDP,
                                  rm::POMDP,
                                  b,
                                  a,
                                  o,
                                  rng::AbstractRNG)
    # we first update the external component of the
    ps = [compose_state(o, internal(p)) for p in particles(bp)]
    bp = WeightedParticleBelief(ps, weights(bp))
    # now we can use low variance resampling to do the rest of the job
    return resample(rs.lv, bp, rng)
end

mutable struct SharedExternalStateFilter{PM,RM,RNG<:AbstractRNG,PMEM} <: Updater
    predict_model::PM
    reweight_model::RM
    resampler::SharedExternalStateResampler # TODO: Maybe people want custom resamplers here
    n_init::Int
    rng::RNG
    _particle_memory::PMEM
    _weight_memory::Vector{Float64}
end

## Constructors ##
function SharedExternalStateFilter(model::POMDP, n::Integer,; rng::AbstractRNG=Random.GLOBAL_RNG)
    return SharedExternalStateFilter(model,
                                     model,
                                     n,
                                     rng=rng)
end

function SharedExternalStateFilter(pmodel, rmodel, n::Integer; rng::AbstractRNG=Random.GLOBAL_RNG)
    return SharedExternalStateFilter(pmodel,
                                     rmodel,
                                     SharedExternalStateResampler(n),
                                     n,
                                     rng,
                                     particle_memory(pmodel),
                                     Float64[]
                                    )
end

function ParticleFilters.initialize_belief(up::SharedExternalStateFilter, distribution)
    particles = [rand(up.rng, distribution) for i in 1:up.n_init]
    first_state = first(particles)
    @assert all(isequal(external(first_state), external(p)) for p in particles)
    return ParticleCollection(particles)
end

function ParticleFilters.update(up::SharedExternalStateFilter, b::ParticleCollection, a, o)
    pm = up._particle_memory
    wm = up._weight_memory
    resize!(pm, n_particles(b))
    resize!(wm, n_particles(b))
    predict!(pm, up.predict_model, b, a, o, up.rng)
    reweight!(wm, up.reweight_model, b, a, pm, o, up.rng)

    return resample(up.resampler,
                    SharedExternalStateBelief{ext_int_typesig(b)...}(pm, wm),
                    up.predict_model,
                    up.reweight_model,
                    b, a, o,
                    up.rng)
end
