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
	Gadfly.set_default_plot_size(20cm,20cm)

	scatter = plot(data, x=:median_planner_time, y=:discounted_reward, color=:planner_hbm_key, Geom.point, Geom.errorbar)

	planner_metrics = extract_value_compute(data)
	df = DataFrame(Model=String[], MeanValue=Float64[], StdErrorValue=Float64[], MeanCompute=Float64[], StdErrorCompute=Float64[])
	for planner_type in keys(planner_metrics)
		(value, compute) = planner_metrics[planner_type]
		push!(df, (planner_type, mean(value), std(value), mean(compute), std(compute)))
	end

	value = plot(x=df.Model, y=df.MeanValue, ymin=(df.MeanValue - df.StdErrorValue),
		 		 ymax=(df.MeanValue + df.StdErrorValue), Geom.point, Geom.errorbar,
		 		 Guide.xlabel("Planner Model"), Guide.ylabel("Value"))
	compute = plot(x=df.Model, y=df.MeanCompute, ymin=(df.MeanCompute - df.StdErrorCompute),
				   ymax=(df.MeanCompute + df.StdErrorCompute), Geom.point, Geom.errorbar,
				   Guide.xlabel("Planner Model"), Guide.ylabel("Computation Time")))
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