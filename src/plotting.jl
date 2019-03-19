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

	scatter = plot(data, x=:median_planner_time, y=:discounted_reward, color=:planner_hbm_key, Geom.point, Geom.errorbar)
	value = plot(data, x=:planner_hbm_key, y=:discounted_reward, Geom.violin, Gadfly.Theme(minor_label_font_size=8pt))
	compute = plot(data, x=:planner_hbm_key, y=:median_planner_time, Geom.violin, Gadfly.Theme(minor_label_font_size=8pt))

	display(Gadfly.title(vstack(scatter, hstack(value, compute)), "$(first(data[:pi_key])) $(first(data[:simulation_hbm_key]))"))
end

function plot_full(datas...)
	data = vcat(datas...)
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
