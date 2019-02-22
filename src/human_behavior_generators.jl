function generate(rng::AbstractRNG, hbg::HumanBehaviorGenerator, m::HSModel)
  hb = rand(rng, hbg.behaviors)
  return generate(rng, hb, m)
end

# TODO: this makes implicit assumptions about ranges. Maybe move to model
generate(rng::AbstractRNG, ::Type{HumanConstantVelocityBehavior}, ::HSModel) = HumanConstantVelocityBehavior(rand(rng))
generate(rng::AbstractRNG, ::Type{HumanPIDBehavior}, m::HSModel) = HumanPIDBehavior(human_target=rand(rng, corner_poses(room(m))))
