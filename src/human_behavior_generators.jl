function generate_human_behavior(rng::AbstractRNG, m::HSModel)
  hb = rand(rng, human_behavior_generator(m).behaviors)
  return generate_human_behavior(rng, hb, m)
end

# TODO: this makes implicit assumptions about ranges. Maybe move to model
generate_human_behavior(rng::AbstractRNG, ::Type{HumanConstantVelocityBehavior}, ::HSModel) = HumanConstantVelocityBehavior(rand(rng))
generate_human_behavior(rng::AbstractRNG, ::Type{HumanPIDBehavior}, m::HSModel) = HumanPIDBehavior(human_target=rand(rng, corner_poses(room(m))))
