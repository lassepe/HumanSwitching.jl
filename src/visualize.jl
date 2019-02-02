"""
room node

Composes a room node for the visualization graph

Fields:
- `rr` a representation of the room
"""
function room_node(rr::RoomRep; fill_color="bisque", stroke_color="black")::Context
  compose(context(), fill(fill_color), stroke(stroke_color),
          rectangle(0, rr.height, rr.width, rr.height))
end
"""
agent_node

Composes an agent node for the visualization graph

Fields:
- `as` the state of the agent to be rendered
- `r` the visual radius of the agent
"""
function agent_node(as::AgentState; r=0.15, fill_color="tomato", stroke_color="black")::Context
  compose(context(), fill(fill_color), stroke(stroke_color),
          (context(), circle(as.xy[1], as.xy[2], r)),
          (context(), line([(as.xy[1], as.xy[2]), (as.xy[1]+cos(as.phi)*r*2, as.xy[2]+sin(as.phi)*r*2)]), linewidth(1)))
end

"""
target_node

Composes a target node (states that agents want to reach) for the visualization graph

Fields:
- `ts` the target state to be visualized
"""
function target_node(ts::AgentState; size=0.15, fill_color="deepskyblue", stroke_color="black")::Context
  compose(context(), fill(fill_color), stroke(stroke_color),
          circle(ts.xy[1], ts.xy[ 2], size/2))
end


# TODO: Reduce code duplication!
function render_scene(room::RoomRep, robot_state::AgentState, human_states::Array{AgentState})
  # place mirror all children along the middle axis of the unit contexnet
  mirror = context(mirror=Mirror(0, 0.5, 0.5))
  # scale all children to fit into the mirrored unit context
  base_scale = context(0, 0, 1/room.width, 1/room.height)

  room_viz = room_node(room)
  human_vizs = [agent_node(h, fill_color="tomato") for h in human_states]
  robot_viz = agent_node(robot_state, fill_color="light blue")

  composition = compose(mirror, (base_scale, human_vizs..., target_viz, robot_viz, room_viz))
  composition |> SVG("display.svg", 14cm, 14cm)
  return composition
end

function render_scene(m::HSModel, s::HSState)
  # extract the room prepresentation from the problem
  room_rep::RoomRep = room(m)
  # place mirror all children along the middle axis of the unit context
  mirror = context(mirror=Mirror(0, 0.5, 0.5))
  # scale all children to fit into the mirrored unit context
  base_scale = context(0, 0, 1/room_rep.width, 1/room_rep.height)

  room_viz = room_node(room_rep)
  human_pose_viz = agent_node(s.human_pose, fill_color="tomato")
  human_target_viz = target_node(s.human_target)

  p_start = [Tuple(s.human_pose.xy)]
  p_end = [Tuple(s.human_target.xy)]

  # TODO MOVE!
  as = s.human_pose
  at = s.human_target
  r = 0.5
  c1 = [(as.xy[1]+cos(as.phi)*r*2, as.xy[2]+sin(as.phi)*r*2)]
  c2 = [(at.xy[1]-cos(at.phi)*r*2, at.xy[2]-sin(at.phi)*r*2)]
  line_to_target = compose(context(), fill("black"), stroke("black"), curve(p_start, c1, c2, p_end))

  composition = compose(mirror, (base_scale, human_pose_viz, human_target_viz, line_to_target, room_viz))
  composition |> SVG("display.svg", 14cm, 14cm)
  return composition
end
