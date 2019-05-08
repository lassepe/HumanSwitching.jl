function process_data_span(data::DataFrame)
    # Postprocess dataframes
    data_POMCP = data[occursin.("POMCP_", data[:solver_setup_key]), :]
    data_POMCPOW = data[occursin.("POMCPOW_", data[:solver_setup_key]), :]
    data_DESPOT = data[occursin.("DESPOT_", data[:solver_setup_key]), :]
    
    md_POMCP = DataFrame(Depth=Int64[], UCB_criterion=Float64[], Max_time=Float64[])
    for i in 1:size(data_POMCP,1)
        solver_type = data_POMCP[:solver_setup_key][i]
        solver_pars = split(solver_type, "_")
        push!(md_POMCP, (parse(Int64, solver_pars[2]), parse(Float64, solver_pars[3]), parse(Float64, solver_pars[4])))
    end
    md_POMCPOW = DataFrame(Depth=Int64[], UCB_criterion=Float64[], Max_time=Float64[])
    for i in 1:size(data_POMCPOW,1)
        solver_type = data_POMCPOW[:solver_setup_key][i]
        solver_pars = split(solver_type, "_")
        push!(md_POMCPOW,(parse(Int64, solver_pars[2]), parse(Float64, solver_pars[3]), parse(Float64, solver_pars[4])))
    end
    md_DESPOT = DataFrame(Scenarios=Int64[], Depth=Int64[], Lambda=Float64[], Max_time=Float64[])
    for i in 1:size(data_DESPOT,1)
        solver_type = data_DESPOT[:solver_setup_key][i]
        solver_pars = split(solver_type, "_")
        push!(md_DESPOT, (parse(Int64, solver_pars[2]), parse(Int64, solver_pars[3]), parse(Float64, solver_pars[4]), parse(Float64, solver_pars[5])))
    end

    data_POMCP = hcat(data_POMCP, md_POMCP)
    data_POMCPOW = hcat(data_POMCPOW, md_POMCPOW)
    data_DESPOT = hcat(data_DESPOT, md_DESPOT)
    return data_POMCP, data_POMCPOW, data_DESPOT
end

function plot_POMCP(data::DataFrame)
    Gadfly.set_default_plot_size(30cm,30cm)
	detailed_theme = Gadfly.Theme(minor_label_font_size=8pt, key_position=:none)
    
    depth = plot(data, x=:Depth, y=:normalized_discounted_reward, Geom.point)
    c = plot(data, x=:UCB_criterion, y=:normalized_discounted_reward, Geom.point)
    time = plot(data, x=:Max_time, y=:normalized_discounted_reward, Geom.point)
    
    final_plot = Gadfly.title(hstack(depth, c, time), 
                             """
                        	 Problem Instance: $(first(data[:pi_key]))
                         	 True Human Model: $(first(data[:simulation_hbm_key]))
                             """)
    display(final_plot)
end

function plot_POMCPOW(data::DataFrame)
    Gadfly.set_default_plot_size(30cm,30cm)
	detailed_theme = Gadfly.Theme(minor_label_font_size=8pt, key_position=:none)
    
    depth = plot(data, x=:Depth, y=:normalized_discounted_reward, Geom.point)
    c = plot(data, x=:UCB_criterion, y=:normalized_discounted_reward, Geom.point)
    time = plot(data, x=:Max_time, y=:normalized_discounted_reward, Geom.point)
    
    final_plot = Gadfly.title(hstack(depth, c, time), 
                             """
                        	 Problem Instance: $(first(data[:pi_key]))
                         	 True Human Model: $(first(data[:simulation_hbm_key]))
                             """)
    display(final_plot)
end

function plot_DESPOT(data::DataFrame)
    Gadfly.set_default_plot_size(30cm,30cm)
	detailed_theme = Gadfly.Theme(minor_label_font_size=8pt, key_position=:none)
   
    # Plot K scenarios vs Value
    k = plot(data, x=:Scenarios, y=:normalized_discounted_reward, Geom.point)

    # Plot Depth vs Value
    depth = plot(data, x=:Depth, y=:normalized_discounted_reward, Geom.point)

    # Plot Lambda vs Value
    lambda = plot(data, x=:Lambda, y=:normalized_discounted_reward, Geom.point)

    # Plot Time vs Value
    time = plot(data, x=:Max_time, y=:normalized_discounted_reward, Geom.point)
    
    final_plot = Gadfly.title(hstack(k, depth, lambda, time), 
                             """
                        	 Problem Instance: $(first(data[:pi_key]))
                         	 True Human Model: $(first(data[:simulation_hbm_key]))
                             """)
    display(final_plot)
end

function plot_solver_comparison(data::DataFrame)
    data_POMCP, data_POMCPOW, data_DESPOT = process_data_span(data)
    D_POMCP, C_POMCP, T_POMCP = plot_POMCP(data_POMCP)
    D_POMCPOW, C_POMCPOW, T_POMCPOW = plot_POMCPOW(data_POMCPOW)
    K_DESPOT, D_DESPOT, L_DESPOT, T_DESPOT = plot_DESPOT(data_DESPOT)
    final_plot = Gadfly.title(vstack(hstack(D_POMCP, C_POMCP, T_POMCP), 
                                     hstack(D_POMCPOW, C_POMCPOW, T_POMCPOW),
                                     hstack(K_DESPOT, D_DESPOT, L_DESPOT, T_DESPOT)),
                             """
                        	 Problem Instance: $(first(data[:pi_key]))
                         	 True Human Model: $(first(data[:simulation_hbm_key]))
                             """)
    display(final_plot)
end

function get_value_SEM(data::DataFrame, field, xlabel::String)
    dt = typeof(first(data[Symbol(field)]))
    df = DataFrame(ParameterValue=dt[], MeanValue=Float64[], SEMValue=Float64[])
    for field_type in unique(data[Symbol(field)])
        filtered_data = data[data[Symbol(field)] .== field_type, :]
        value = filtered_data[:normalized_discounted_reward]
        push!(df, (field_type, mean(value), std(value)/sqrt(size(data,1))))
    end

    return plot(x=df.ParameterValue, y=df.MeanValue, ymin=(df.MeanValue - df.SEMValue), ymax=(df.MeanValue + df.SEMValue),
                color=df.ParameterValue, Geom.point, Geom.errorbar, 
                Guide.xlabel(xlabel), Guide.ylabel("Value"))
end

function plot_points(data::DataFrame)
	Gadfly.set_default_plot_size(30cm,30cm)
	detailed_theme = Gadfly.Theme(minor_label_font_size=8pt, key_position=:none)

	df = DataFrame(Model=String[], MeanValue=Float64[], SEMValue=Float64[], MeanCompute=Float64[], SEMCompute=Float64[])
	for planner_type in unique(data.planner_hbm_key)
		for solver_type in unique(data.solver_setup_key)
			common_rows = data[(data[:planner_hbm_key] .== planner_type).&(data[:solver_setup_key] .== solver_type), :]
			value = common_rows[:normalized_discounted_reward]
			compute = common_rows[:combined_median_time]
			push!(df, (planner_type.*solver_type, mean(value), std(value)/sqrt(size(common_rows,1)), mean(compute), std(compute)/sqrt(size(common_rows,1))))
		end
	end

	value_v_compute = plot(x=df.MeanCompute, y=df.MeanValue,
		     	       xmin=(df.MeanCompute - df.SEMCompute), xmax=(df.MeanCompute + df.SEMCompute),
		     	       ymin=(df.MeanValue - df.SEMValue), ymax=(df.MeanValue + df.SEMValue),
		     	       color=df.Model, Geom.point, Geom.errorbar, Guide.xlabel("Compute"), Guide.ylabel("Value"))

	scatter = plot(data, x=:combined_median_time, y=:normalized_discounted_reward, color=(data.planner_hbm_key.*data.solver_setup_key), Geom.point)

	success_rate =  plot(data, xgroup=:planner_hbm_key, x=:final_state_type, color=:planner_hbm_key, Geom.subplot_grid(Geom.histogram),
                         Gadfly.Theme(major_label_font_size=8pt, minor_label_font_size=8pt, key_position=:none))

	final_plot = Gadfly.title(vstack(value_v_compute, scatter),
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
    non_terminal_data = @linq data |> where(:final_state_type .== "nonterminal")
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
        push!(data_frames, transform_data(df; shorten_names=shorten_names))
    end
    # stack them to one long table
    all_data = vcat(data_frames...)

    return all_data
end

function transform_data(data::DataFrame; shorten_names::Bool=true)
    modified_data = !shorten_names ? data : @linq data |> transform(planner_hbm_key=simplify_hbm_name.(:planner_hbm_key),
                                                                            simulation_hbm_key=simplify_hbm_name.(:simulation_hbm_key))

    modified_data[:combined_median_time] = modified_data[:median_updater_time] .+ modified_data[:median_prediction_time] .+ modified_data[:median_planning_time]
    modified_data[:normalized_discounted_reward] = modified_data[:discounted_reward] .- modified_data[:free_space_estimate]

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

filter_by_planner(data::DataFrame, s::String) = data[occursin.(s, data.planner_hbm_key), :]

function tail_expectation(vals::Array, q::Float64)
    @assert 0 <= q <= 1
    n_lower_q_vals::Int = floor(q * length(vals))
    return mean(sort(vals)[1:n_lower_q_vals])
end

function statistics(data::DataFrame)
    for p in unique(data.planner_hbm_key)
        planner_data = filter_by_planner(data, p)
        println("""
                ---------------------------------------------------------------------------------

                Planner: $p

                tail_expectation: $(tail_expectation(planner_data.normalized_discounted_reward, 0.20))
                mean: $(mean(planner_data.normalized_discounted_reward))
                lcb: $(mean(planner_data.normalized_discounted_reward) - std(planner_data.normalized_discounted_reward))
                """)
    end
end
