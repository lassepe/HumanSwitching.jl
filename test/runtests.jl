using Test
using Suppressor

using HumanSwitching
const HS = HumanSwitching

using Random
using LinearAlgebra
using Statistics

using BeliefUpdaters
using POMDPs
using POMDPPolicies
using POMDPSimulators
using POMDPGifs

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
  phuman_mdp = HSMDP(transition_model=PControlledHumanTransition())
  hs_pomdp_exact_o = HSPOMDP(sensor=ExactPositionSensor(), mdp=phuman_mdp)
  hs_pomdp_noisy_o = HSPOMDP(sensor=NoisyPositionSensor([0.001,0.001,0.01]), mdp=phuman_mdp)
  s = initialstate(hs_pomdp_exact_o, rng)
  a = HS.HSAction()
  # Transition model, simply checking whether the call is successfull
  sp = HS.generate_s(hs_pomdp_exact_o, s, a, rng)

  # Obsevation model:
  # the deterministic observation model
  @test HS.generate_o(hs_pomdp_exact_o, s, a, sp, rng) == sp.human_pose
  # the noisy obsevation model
  test_obs_data = collect(HS.generate_o(hs_pomdp_noisy_o, s, a, sp, rng) for i in 1:5)
  dist = norm(mean(test_obs_data) - sp.human_pose)
  @test 0 <= dist <= 0.1

  # Initial state generation
  test_inits_data = [HS.initialstate(hs_pomdp_exact_o, rng) for i in 1:10000]
  r = HS.room(hs_pomdp_exact_o)
  @test all(HS.isinroom(td.human_pose, r) && HS.isinroom(td.human_target, r) for td in test_inits_data)

  # check whether the simulation terminates in finite time if we only observe
  s = initialstate(hs_pomdp_noisy_o, rng)
  policy = FunctionPolicy(x->HSAction())
  belief_updater = NothingUpdater()
  history = simulate(HistoryRecorder(max_steps=500), hs_pomdp_noisy_o, policy, belief_updater)
  # note that only sp is terminal, not s! (you never take an action from the
  # terminal state)
  last_s = last(collect(sp for sp in eachstep(history, "sp")))
  @test_broken isterminal(hs_pomdp_noisy_o, last_s)
end;

# this test set checks whether everything is implemented to be pseudo-random.
# Meaning that with the same rng we should get the same result!
@testset "POMDP deterministic checks" begin
  mdp = HSMDP(transition_model=PControlledHumanAWGNTransition())
  pomdp = HSPOMDP(sensor=NoisyPositionSensor(), mdp=mdp)
  a = HS.HSAction()

  # ORDER of everything matters (rng1 and rng2 must undergo the same
  # "transitions")
  rng1 = MersenneTwister(42)
  rng2 = MersenneTwister(42)

  # Initial states:
  s1 = initialstate(pomdp, rng1)
  s2 = initialstate(pomdp, rng2)
  @test isequal(s1, s2)

  # Transitions
  sp1 = generate_s(pomdp, s1, a, rng1)
  sp2 = generate_s(pomdp, s2, a, rng2)
  @test isequal(sp1, sp2)

  # Observation
  o1 = generate_o(pomdp, s1, a, sp1, rng1)
  o2 = generate_o(pomdp, s2, a, sp2, rng2)
  @test isequal(o1, o2)
end;

@testset "POMDP visualization" begin
  mdp = HSMDP(transition_model=PControlledHumanTransition())
  pomdp = HSPOMDP(sensor=NoisyPositionSensor([0.1,0.1,0.01]), mdp=mdp)
  rng = MersenneTwister(42)
  belief_updater = NothingUpdater()
  policy = FunctionPolicy(x->HSAction())
  # this only checks wether the calls work without error (@test_nowarn doesn't
  # work due to progressbar I assume)
  @test_nowarn makegif(pomdp, policy, belief_updater, filename=joinpath(@__DIR__, "test_renderings", "makegif_test.gif"), rng=rng, max_steps=100, show_progress=false)
end;
