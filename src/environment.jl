struct Room
  floor::Rectangle
end

function render(ctx::CairoContext, r::Rectangle; rgb_color=(0.8, 0.8, 0.8))
  save(ctx)
  set_source_rgb(ctx, rgb_color...);    # light gray

  # get transformed coordinates of the recatangle
  top_left_x, top_left_y = transform_coords(r.top_left)
  rectangle(ctx, top_left_x, top_left_y, transform_scale(r.width), transform_scale(r.height)); # background

  fill(ctx);
  restore(ctx)
end
