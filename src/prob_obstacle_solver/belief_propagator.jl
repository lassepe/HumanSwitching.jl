"""
    BeliefPropagator

"""
abstract type BeliefPropagator end
"""
    initialize_belief(bp::BeliefPropagator, b)

initializes the internal belief representation from a given belief
"""
function initialize_belief end

"""
    predict(m, b, rng)

Simulate each of the particles in `b` forward one time step using model `m`, returning a vector of states.

This function is provided for convenience only. New models should implement `predict!`.
"""
function predict end

"""
    predict!

Same as `predict`, but with `pm` being a particle memory to reduce expensive heap allocations.
"""
function predict! end

@with_kw struct ParticleBeliefPropagator{PM, RNG<:AbstractRNG, PMEM} <: BeliefPropagator
    predict_model::PM
    n_init::Int
    rng::RNG
    _pm::PMEM = particle_memory(predict_model)
end

function initialize_belief(bp::ParticleBeliefPropagator, b::ParticleCollection)
    # if we already have the right number of pacrticles, we can simply keep the belief
    return n_particles(b) == bp.n_init ? b : resample(LowVarianceResampler(bp.n_init), b, bp.rng)
end

function predict!(pm, m::PredictModel, b::ParticleCollection, rng::AbstractRNG)
    for i in 1:n_particles(b)
        x1 = particle(b, i)
        pm[i] = m.f(x1, rng)
    end
end

function predict(m, b, rng)
    pm = particle_memory(m)
    resize!(pm, n_particles(b))
    predict!(pm, m, b, args...)
    return pm
end

predict(bp::ParticleBeliefPropagator, b) = predict(bp.predict_model, b, bp.rng)
