mutable struct BehaviorParticleBelief{G}
  " the fully observable part of the state (poses of all agents and robot target) "
  external::HSExternalState
  " particles to keep track of the hidden part of the state (hidden: HumanBehaviorModel) "
  particles::Vector{HumanBehaviorModel}
  " culative weights "
  cweights::Vector{Float64}
end

"""
Return the weights of the behavior particles (as oppose to the cumulative weights)
"""
weights(b::BehaviorParticleBelief) = insert!(diff(b.cweights), 1, first(b.cweights))

"""
Use bisection search to sample efficiently given a vector of objects and the cumulative weights.
"""
function sample_cweighted(rng::AbstractRNG, objects::AbstractVector, cweights::AbstractVector{Float64})
  t = rand(rng)*cweights[end]
  large = length(cweights) # index of cdf value that is bigger than t
  small = 0 # index of cdf value that is smaller than t
  while large > small + 1
    new = div(small + large, 2)
    if t < cweights[new]
      large = new
    else
      small = new
    end
  end
  return objects[large]
end

# function rand(rng::AbstractRNG,
#               b::BehaviorParticleBelief,
#               s::HSState=HSState(external=b.external, ))

function most_likely_state(b::BehaviorParticleBelief)
  human_behavior_model = b.particles[argmax(weights(b))]
  return HSState(external=b.ext external, hbm=human_behavior_model)
end

function  
