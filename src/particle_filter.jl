mutable struct BehaviorParticleBelief{G}
  " the fully observable part of the state (poses of all agents and robot target) "
  external_state::HSExternalState
  " particles to keep track of the hidden part of the state (hidden: HumanBehaviorModel) "
  particles::Vector{HumanBehaviorModel}
  " culative weights "
  cweights::Vector{Float64}
end

