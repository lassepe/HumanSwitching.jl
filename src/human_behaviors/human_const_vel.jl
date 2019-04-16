"""
HumanConstVelBehavior
"""
struct HumanConstVelBState <: HumanBehaviorState
    vx::Float64
    vy::Float64
end

free_evolution(hbs::HumanConstVelBState, p::Pos) = Pos(p.x + dt*hbs.vx, p.y + dt*hbs.vy)

@with_kw struct HumanConstVelBehavior <: HumanBehaviorModel
    vel_max::Float64 = 1.4
    vel_resample_sigma::Float64 = 0.0
end

bstate_type(::HumanConstVelBehavior)::Type = HumanConstVelBState

# this model randomely generates HumanConstVelBState from the min_max_vel range
rand_hbs(rng::AbstractRNG, hbm::HumanConstVelBehavior) = HumanConstVelBState(rand(rng, Uniform(-hbm.vel_max, hbm.vel_max)),
                                                                             rand(rng, Uniform(-hbm.vel_max, hbm.vel_max)))

function human_transition(hbs::HumanConstVelBState, hbm::HumanConstVelBehavior, m::HSModel,
                          p::Pos, rng::AbstractRNG)
    human_pos_p = free_evolution(hbs, p)
    hbs_p = (!iszero(hbm.vel_resample_sigma) ? HumanConstVelBState(hbs.vx + randn(rng)*hbm.vel_resample_sigma,
                                                                   hbs.vy + randn(rng)*hbm.vel_resample_sigma) :
             hbs)

    return human_pos_p, hbs_p
end

