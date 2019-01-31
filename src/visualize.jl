function room_node(rr::RoomRep; fill_color="bisque", stroke_color="black")::Context
  compose(context(), fill(fill_color), stroke(stroke_color),
          rectangle(0, rr.height, rr.width, rr.height))
end

function agent_node(as::AgentState; r=0.15, fill_color="tomato", stroke_color="black")::Context
  compose(context(), fill(fill_color), stroke(stroke_color),
          (context(), circle(as.xy[1], as.xy[2], r)),
          (context(), line([(as.xy[1], as.xy[2]), (as.xy[1]+cos(as.phi)*r*2, as.xy[2]+sin(as.phi)*r*2)]), linewidth(1)))
end

function render_scene(room::RoomRep, robot_state::AgentState, human_states::Array{AgentState})
  # place mirror all children along the middle axis of the unit context
  mirror = context(mirror=Mirror(0, 0.5, 0.5))
  # scale all children to fit into the mirrored unit context
  base_scale = context(0, 0, 1/room.width, 1/room.height)

  room_viz = room_node(room)
  human_vizs = [agent_node(h, fill_color="tomato") for h in human_states]
  robot_viz = agent_node(robot_state, fill_color="light blue")

  composition = compose(mirror, (base_scale, human_vizs..., robot_viz, room_viz))
  composition |> SVG("display.svg", 14cm, 14cm)
  return composition
end
