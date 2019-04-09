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

    scatter = plot(data, x=:total_median_cpu_time, y=:discounted_reward, color=:planner_hbm_key, Geom.point)

    violin_plot_appearance = (Geom.violin, Gadfly.Theme(minor_label_font_size=8pt, key_position=:none))
    value = plot(data, x=:planner_hbm_key, y=:discounted_reward, color=:planner_hbm_key, violin_plot_appearance...)
    compute = plot(data, x=:planner_hbm_key, y=:total_median_cpu_time, color=:planner_hbm_key, violin_plot_appearance...)

    success_rate =  plot(data, xgroup=:planner_hbm_key, x=:final_state_type, color=:planner_hbm_key, Geom.subplot_grid(Geom.histogram),
                         Gadfly.Theme(major_label_font_size=8pt, minor_label_font_size=8pt, key_position=:none))

    final_plot = Gadfly.title(vstack(scatter,
                                hstack(value, compute),
                                success_rate),
                         """
                         Problem Instance: $(first(data[:pi_key]))
                         True Human Model: $(first(data[:simulation_hbm_key]))
                         """)

	display(final_plot)
end

simplify_hbm_name(s::String) = string(split(s, "HumanMultiGoalBoltzmann")...)

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

function check_data(data::DataFrame)
    non_terminal_data = @linq data |> where(:final_state_type .== "non-terminal")
    if nrow(non_terminal_data) > 0
        @warn "There were non-terminal runs:"
        display(non_terminal_data)
        return false
    else
        @assert count(data.final_state_type .== "success") + count(data.final_state_type .== "failure") == nrow(data)
        @info "Checks on data succeeded!"
        return true
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


    modified_data = !shorten_names ? all_data : @linq all_data |> transform(planner_hbm_key=simplify_hbm_name.(:planner_hbm_key),
                                                                            simulation_hbm_key=simplify_hbm_name.(:simulation_hbm_key))

    modified_data[:total_median_cpu_time] = modified_data[:median_updater_time] .+ modified_data[:median_planner_time]

    # sanity check the data
    check_data(modified_data)

    return modified_data
end

# TODO: this should not filter. Instead, plot the success rate per planer_hbm_key!
function success_rate(planner_hbm_key::String, all_data::DataFrame)
    filtered_data = @linq all_data |> where(:planner_hbm_key .== planner_hbm_key)
    n_total = nrow(filtered_data)
    n_success = nrow((@linq filtered_data |> where(:final_state_type .== "success")))
    return n_success / n_total
end
