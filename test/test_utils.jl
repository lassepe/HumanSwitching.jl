@testset "normalized_angle_diff" begin
    @test isapprox(HS.normalized_angle_diff(pi/2), pi/2)
    @test isapprox(HS.normalized_angle_diff(-pi/2), -pi/2)
    @test isapprox(HS.normalized_angle_diff(2pi), 0)
    @test isapprox(HS.normalized_angle_diff(1.5pi), -0.5pi)
    @test isapprox(HS.normalized_angle_diff(Float64(pi)), pi)
end;

