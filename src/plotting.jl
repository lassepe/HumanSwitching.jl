function extract_value_compute(data::DataFrame)
	planner_models = unique(data[:planner_hbm_key])
	planner_metrics = Dict{String,Tuple{Array, Array}}()
	for planner_type in planner_models
		relevant_rows = data[data[:planner_hbm_key] .== planner_type, :]
		value = relevant_rows[:discounted_reward]
		compute = relevant_rows[:median_planner_time]
		planner_metrics[planner_type] = (value, compute)
	end
	return planner_metrics
end

function plot_points(data::DataFrame)
	Gadfly.set_default_plot_size(30cm,30cm)

	scatter = plot(data, x=:median_planner_time, y=:discounted_reward, color=:planner_hbm_key, Geom.point)

    violin_plot_appearance = (Geom.violin, Gadfly.Theme(minor_label_font_size=8pt, key_position=:none))
    value = plot(data, x=:planner_hbm_key, y=:discounted_reward, color=:planner_hbm_key, violin_plot_appearance...)
	compute = plot(data, x=:planner_hbm_key, y=:median_planner_time, color=:planner_hbm_key, violin_plot_appearance...)

    success_rate =  plot(data, xgroup=:planner_hbm_key, x=:final_state_type, Geom.subplot_grid(Geom.histogram(density=true)),
                         Gadfly.Theme(major_label_font_size=8pt, minor_label_font_size=8pt, key_position=:none))

	display(Gadfly.title(vstack(scatter,
                                hstack(value, compute),
                                success_rate),
                         """
                         Problem Instance: $(first(data[:pi_key]))
                         True Human Model: $(first(data[:simulation_hbm_key]))
                         """)
           )
end

simplify_hbm_name(s::String) = string(split(s, "Human")...)

function plot_full(data)
	problem_instances = unique(data[:pi_key])
	for pi_type in problem_instances
		println("Plotting problem instance: $pi_type")
		plot_problem_instance(data[data[:pi_key] .== pi_type, :])
	end
end

function plot_problem_instance(data::DataFrame)
	simulation_models = unique(data[:simulation_hbm_key])
	for simulation_type in simulation_models
		println("Plotting simulation type: $simulation_type")
		plot_points(data[data[:simulation_hbm_key] .== simulation_type, :])
	end
end

function load_data(files...; shorten_names::Bool=true)
    data_frames = []

    # load all files
    for file in files
        type_hints = Dict(:hist_validation_hash=>String)
        df = CSV.read(file, types=type_hints)
        push!(data_frames, df)
    end
    # stack them to one long table
    all_data = vcat(data_frames...)

    return !shorten_names ? all_data : @linq all_data |> transform(planner_hbm_key=simplify_hbm_name.(:planner_hbm_key),
                                                                   simulation_hbm_key=simplify_hbm_name.(:simulation_hbm_key))
end

