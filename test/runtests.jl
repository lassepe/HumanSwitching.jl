using Test

using HumanSwitching
const HS = HumanSwitching

using Random
using LinearAlgebra
using Statistics

@testset "normalized_angle_diff" begin
  @test isapprox(HS.normalized_angle_diff(pi/2), pi/2)
  @test isapprox(HS.normalized_angle_diff(-pi/2), -pi/2)
  @test isapprox(HS.normalized_angle_diff(2pi), 0)
  @test isapprox(HS.normalized_angle_diff(1.5pi), -0.5pi)
  @test isapprox(HS.normalized_angle_diff(Float64(pi)), pi)
end;

@testset "POMDP interface" begin
  # checking whether we can actually succesfully construct all those types
  rng = MersenneTwister(42)
  hs_pomdp_exact = HS.HSPOMDP(HS.ExactPositionSensor())
  hs_pomdp_noisy = HS.HSPOMDP(HS.NoisyPositionSensor([0.1,0.1,0.01]))
  s = initialstate(hs_pomdp_exact, rng)
  a = HS.HSAction()
  # Transition model, simply checking whether the call is successfull
  sp = HS.generate_s(hs_pomdp_exact, s, a, rng)
  ## TODO: check reproducabiliy (same result with same initial state and same rng)

  # Obsevation model:
  # the deterministic observation model
  @test HS.generate_o(hs_pomdp_exact, s, a, sp, rng) == sp.human_pose
  # the noisy obsevation model
  test_obs_data = collect(HS.generate_o(hs_pomdp_noisy, s, a, sp, rng).xy for i in 1:10000)
  dist = norm(mean(test_obs_data) - sp.human_pose.xy)
  @test 0 <= dist <= 0.1

  # Initial state generation
  test_inits_data = [HS.initialstate(hs_pomdp_exact, rng) for i in 1:10000]
  r = HS.room(hs_pomdp_exact)
  @test all(HS.isinroom(td.human_pose, r) && HS.isinroom(td.human_target, r) for td in test_inits_data)
end;
