"""
HSParticleFilter

Definition of the particle filter for the HumanSwitching problem

Fields:
- `pf::SIRParticleFilter` a SIRParticleFilter as defined in `ParticleFilters.jl`
- `sensor::Union{ExactPositionSensor, NoisyPositionSensor}` the sensor to be used
"""

struct HSParticleFilter <: POMDPs.Updater
  spf::SIRParticleFilter
  sensor::Union{ExactPositionSensor, NoisyPositionSensor}
  lvr::LowVarianceResampler
end

struct HSPositionSensorResampler
  n::Int64 # number of particles to be maintained
end

# TODO: reason about whether we could even get rid of the custom resampler here
# if it only forwards anyway
ParticleFilters.resample(rs::HSPositionSensorResampler, b::AbstractParticleBelief, rng::AbstractRNG) = resample(  rs.lvr, b, rng)

function POMDPs.update()
