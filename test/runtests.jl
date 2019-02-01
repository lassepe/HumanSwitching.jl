using Test

using HumanSwitching
const HS = HumanSwitching

using Random
using LinearAlgebra

function random_agent_state(range_x::Array{Float64}=[0., 10.], range_y::Array{Float64}=[0., 10.])::AgentState
  x = (rand() * (range_x[2] - range_x[1])) - range_x[1]
  y = (rand() * (range_y[2] - range_y[1])) - range_y[1]
  phi = rand() * pi
  return AgentState(xy=[x, y], phi=phi)
end

@testset "normalized_angle_diff" begin
  @test isapprox(HS.normalized_angle_diff(pi/2), pi/2)
  @test isapprox(HS.normalized_angle_diff(-pi/2), -pi/2)
  @test isapprox(HS.normalized_angle_diff(2pi), 0)
  @test isapprox(HS.normalized_angle_diff(1.5pi), -0.5pi)
  @test isapprox(HS.normalized_angle_diff(Float64(pi)), pi)
end;

@testset "POMDP interface" begin
  # checking whether we can actually succesfully construct all those types
  h_start = random_agent_state()
  h_goal = random_agent_state()
  s = HS.HSState(human_pose=h_start, human_target=h_goal)
  a = HS.HSAct()
  hs_pomdp = HS.HSPOMDP()
  # check whether the current implementation of the human walks in a straight line towards the goal
  s_p = HS.generate_s(hs_pomdp, s, a, MersenneTwister(42))
  @test isapprox(HS.human_angle_to_target(s), HS.human_angle_to_target(s_p))
  @test norm(HS.human_vec_to_target(s_p)) <= norm(HS.human_vec_to_target(s))
end;
