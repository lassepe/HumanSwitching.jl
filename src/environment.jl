# the physical representation of a room
@with_kw struct RoomRep
  width::Float64 = 20
  height::Float64 = 20
end

# the physical representation of an agent
@with_kw struct AgentState
  xy::SVector{2, Float64} = [0, 0]# the x- and y-position
  phi::Float64 = 0 # the orientation of the human
end
