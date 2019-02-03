struct ObserveOnly end
# No matter what the belief is, this policy will only wait and observe
function POMDPs.action(p::ObserveOnly, ::AbstractParticleBelief)::HSAction
  return HSAction()
end

