struct Circle
    "the center point of the circle"
    c::Pos
    "the radius of the circle"
    r::Float64
end
contains(c::Circle, x::Pos) = dist_to_pos(c.c, x) <= c.r

struct Vec3 <: FieldVector{3, Float64}
    x::Float64
    y::Float64
    z::Float64
end

struct InfiniteCone
    "vertex, the tip of the cone"
    v::Vec3
    "direction, the unit length direction of the cone axis (pointing inwards)"
    d::Vec3
    "half opening angle of the cone (measured from the axis to the surface)"
    theta::Float64
    "Inner constructor to enforce certain properties of parameters"
    InfiniteCone(v::Vec3, d::Vec3, theta::Float64) = begin
        @assert isapprox(norm(d), 1)
        @assert 0 < theta < pi/2
        return new(v, d, theta)
    end
end
contains(c::InfiniteCone, x::Vec3) = dot(c.d, (x-c.v)) >= norm(x-c.v) * cos(c.theta)

struct ConicalFrustum
    "the infinite cone version (of which this is a subset)"
    ic::InfiniteCone
    "the distance from the vertex (along the cone axis) at which the frustum starts"
    d1::Float64
    "the distance from the vertex (along the cone axis) at which the frustum ends"
    d2::Float64
    "Inner constructor to enforce certain properties of parameters"
    ConicalFrustum(ic::InfiniteCone, d1::Float64, d2::Float64) = begin
        @assert 0 <= d1 < d2
        new(ic, d1, d2)
    end
end
contains(c::ConicalFrustum, x::Vec3) = contains(c.ic, x) && (c.d1 <= dot(c.ic.d, x) <= c.d2)

struct LineSegment{T<:AbstractVector}
    "The start of the line segment"
    p1::T
    "The end of the line segment"
    p2::T
end
