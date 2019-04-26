"""
HumanConstVelBehavior
"""
struct HumanConstVelBState <: HumanBehaviorState
    vx::Float64
    vy::Float64
end

free_evolution(hbs::HumanConstVelBState, p::Pos) = Pos(p.x + dt*hbs.vx, p.y + dt*hbs.vy)

@with_kw struct HumanConstVelBehavior <: HumanBehaviorModel
    speed_max::Float64 = 1.4
    vel_resample_sigma::Float64 = 0.0
end

speed_max(hbm::HumanConstVelBehavior) = hbm.speed_max

bstate_type(::HumanConstVelBehavior)::Type = HumanConstVelBState

# this model randomely generates HumanConstVelBState with random velocities of
# magnitude up to speed_max
function rand_hbs(rng::AbstractRNG, hbm::HumanConstVelBehavior)
    vx, vy = rand_from_circle(rng, hbm.speed_max)
    return HumanConstVelBState(vx, vy)
end

function human_transition(hbs::HumanConstVelBState, hbm::HumanConstVelBehavior, m::HSModel,
                          p::Pos, rng::AbstractRNG)
    human_pos_p = free_evolution(hbs, p)
    v_p = [hbs.vx + randn(rng) * hbm.vel_resample_sigma,
           hbs.vy + randn(rng) * hbm.vel_resample_sigma]
    speed_p = norm(v_p)
    if speed_p > hbm.speed_max
        # clip to max speed
        v_p = v_p / speed_p * hbm.speed_max
    end
    hbs_p = HumanConstVelBState(v_p[1], v_p[2])

    return human_pos_p, hbs_p
end

