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
