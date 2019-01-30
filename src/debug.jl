using Pkg

if !haskey(Pkg.installed(), "POMotionPlanning")
  # load the environment if not yet done
  jenv = joinpath(dirname(@__FILE__()), "../.")
  Pkg.activate(jenv)
  @info("Activated Environment")
end

using Revise
using POMotionPlanning
using Printf

using Gtk, Cairo, Graphics

function visualize_test()
  # create  a test room
  rect1 = Rectangle(top_left=[0.0, 0.0], width=30.0, height=30.0)
  rect2 = Rectangle(top_left=[0.0, 0.0], width=20.0, height=20.0)
  rect3 = Rectangle(top_left=[0.0, 0.0], width=10.0, height=10.0)
  # create a GTK window and a canvas to render things on:
  canvas = @GtkCanvas
  window = GtkWindow(canvas, "Room")

  @guarded draw(canvas) do widget
    ctx = getgc(canvas)
    h = height(canvas)
    w = width(canvas)

    set_source_rgb(ctx, 0, 0, 0)
    render(ctx, rect1, rgb_color=(1, 0, 0))
    render(ctx, rect2, rgb_color=(0, 1, 0))
    render(ctx, rect3, rgb_color=(0, 0, 1))

    # show some information for debugging
    move_to(ctx, 6/8*w, 1/8*h)
    show_text(ctx, "test")
    stroke(ctx)
  end
  show(canvas)
end

visualize_test()
