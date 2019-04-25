"""
    BeliefPropagator

BeliefPropagators are used for open loop prediction of future belief states,
given an initial belief.
"""
abstract type BeliefPropagator end

"""
    ParticleBeliefPropagator

A BeliefPropagator that uses Monte Carlo integration to approximate the future
belief state.
"""
@with_kw struct ParticleBeliefPropagator{PM, RNG<:AbstractRNG} <: BeliefPropagator
    predict_model::PM
    n_init::Int
    rng::RNG
end

"""
    initialize_belief(bp::BeliefPropagator, b)

initializes the internal belief representation from a given belief
"""
# if we already have the right number of pacrticles, we can simply keep the belief
POMDPs.initialize_belief(bp::ParticleBeliefPropagator, b::ParticleCollection) = n_particles(b) == bp.n_init ? b : resample(LowVarianceResampler(bp.n_init), b, bp.rng)
# there might a b lower variance version of this
POMDPs.initialize_belief(bp::ParticleBeliefPropagator, distribution) = ParticleCollection([rand(bp.rng, distribution) for i in 1:bp.n_init])


"""
    predict!

Same as `predict`, but with `pm` being a particle memory to reduce expensive heap allocations.
"""
function predict!(pm, m::PredictModel, b::ParticleCollection, rng::AbstractRNG)
    for i in 1:n_particles(b)
        x1 = particle(b, i)
        pm[i] = m.f(x1, rng)
    end
end

"""
    predict(m, b, rng)

Simulate each of the particles in `b` forward one time step using model `m`, returning a vector of states.

This function is provided for convenience only. New models should implement `predict!`.
"""
function predict(m, b, rng)
    pm = particle_memory(m)
    resize!(pm, n_particles(b))
    predict!(pm, m, b, rng)
    return pm
end

predict(bp::ParticleBeliefPropagator, b) = predict(bp.predict_model, b, bp.rng)
