solver_parameter_map = Dict{String, Array{String}}(
                                                   "POMCP" => ["Depth", "UCB_criterion", "Time"],
                                                   "POMCPOW" => ["Depth", "UCB_criterion", "Time"],
                                                   "DESPOT" => ["Depth", "Scenarios", "Lambda", "Time"]
                                                  )

function process_data_span(data::DataFrame)
    # Postprocess dataframes
    data_POMCP = data[occursin.("POMCP_", data[:solver_setup_key]), :]
    data_POMCPOW = data[occursin.("POMCPOW_", data[:solver_setup_key]), :]
    data_DESPOT = data[occursin.("DESPOT_", data[:solver_setup_key]), :]
    
    md_POMCP = DataFrame(Depth=Int64[], UCB_criterion=Float64[], Time=Float64[])
    for i in 1:size(data_POMCP,1)
        solver_type = data_POMCP[:solver_setup_key][i]
        solver_pars = split(solver_type, "_")
        push!(md_POMCP, (parse(Int64, solver_pars[2]), parse(Float64, solver_pars[3]), parse(Float64, solver_pars[4])))
    end
    md_POMCPOW = DataFrame(Depth=Int64[], UCB_criterion=Float64[], Time=Float64[])
    for i in 1:size(data_POMCPOW,1)
        solver_type = data_POMCPOW[:solver_setup_key][i]
        solver_pars = split(solver_type, "_")
        push!(md_POMCPOW,(parse(Int64, solver_pars[2]), parse(Float64, solver_pars[3]), parse(Float64, solver_pars[4])))
    end
    md_DESPOT = DataFrame(Depth=Int64[], Scenarios=Int64[], Lambda=Float64[], Time=Float64[])
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
    
    # Plot D/C/T vs Value
    depth = plot_value_SEM(data, "Depth", "Depth of Search Tree", static_fields=true)
    c = plot_value_SEM(data, "UCB_criterion", "UCB_criterion", static_fields=true)
    time = plot_value_SEM(data, "Time", "Maximum planning time", static_fields=true)

    final_plot = Gadfly.title(hstack(depth, c, time), 
                             """
                        	 Problem Instance: $(first(data[:pi_key]))
                         	 True Human Model: $(first(data[:simulation_hbm_key]))
                             Planner Model: $(first(data[:planner_hbm_key]))
                             Solver: POMCP
                             """)
    display(final_plot)
end

function plot_POMCPOW(data::DataFrame)
    Gadfly.set_default_plot_size(30cm,30cm)
	detailed_theme = Gadfly.Theme(minor_label_font_size=8pt, key_position=:none)
    
    # Plot D/C/T vs Value
    depth = plot_value_SEM(data, "Depth", "Depth of Search Tree", static_fields=true)
    c = plot_value_SEM(data, "UCB_criterion", "UCB_criterion", static_fields=true)
    time = plot_value_SEM(data, "Time", "Maximum planning time", static_fields=true)

    final_plot = Gadfly.title(hstack(depth, c, time), 
                             """
                        	 Problem Instance: $(first(data[:pi_key]))
                         	 True Human Model: $(first(data[:simulation_hbm_key]))
                             Planner Model: $(first(data[:planner_hbm_key]))
                             Solver: POMCPOW
                             """)
    display(final_plot)
end

function plot_DESPOT(data::DataFrame)
    Gadfly.set_default_plot_size(30cm,30cm)
	detailed_theme = Gadfly.Theme(minor_label_font_size=8pt, key_position=:none)
   
    # Plot K/D/L/T vs Value
    k = plot_value_SEM(data, "Scenarios", "Number of Sampled Scenarios", static_fields=true)
    depth = plot_value_SEM(data, "Depth", "Depth of Search Tree", static_fields=true)
    lambda = plot_value_SEM(data, "Lambda", "Regularization", static_fields=true)
    time = plot_value_SEM(data, "Time", "Maximum planning time", static_fields=true)

    final_plot = Gadfly.title(hstack(k, depth, lambda, time), 
                             """
                        	 Problem Instance: $(first(data[:pi_key]))
                         	 True Human Model: $(first(data[:simulation_hbm_key]))
                             Planner Model: $(first(data[:planner_hbm_key]))
                             Solver: DESPOT
                             """)
    display(final_plot)
end

function plot_solver_comparison(data::DataFrame)
    data_POMCP, data_POMCPOW, data_DESPOT = process_data_span(data)
    plot_POMCP(data_POMCP)
    plot_POMCPOW(data_POMCPOW)
    plot_DESPOT(data_DESPOT)
    #plot_POMCP_comparison(data_POMCP)
    #plot_POMCPOW_comparison(data_POMCPOW)
    #plot_DESPOT_comparison(data_DESPOT)
end

function best_parameters(data::DataFrame)
    best_value = -10000.0
    best_solver = nothing
    for solver_type in unique(data[:solver_setup_key])
        average_value = mean(data[data[:solver_setup_key] .== solver_type, :].normalized_discounted_reward)
        if average_value > best_value
            best_value = average_value
            best_solver = solver_type
        end
    end
    return best_solver
end

function plot_value_SEM(data::DataFrame, plotting_field, xlabel::String; static_fields::Bool=false)
    dt = typeof(first(data[Symbol(plotting_field)]))
    df = DataFrame(ParameterValue=dt[], MeanValue=Float64[], SEMValue=Float64[])
    if static_fields
        best_params = split(best_parameters(data), "_")
        solver_fields = solver_parameter_map[best_params[1]]
        for (i, field) in enumerate(solver_fields)
            field_type = typeof(first(data[Symbol(field)]))
            if ! (field == plotting_field)
                data = data[data[Symbol(field)] .== parse(field_type, best_params[i+1]), :] 
            end
        end
    end

    for field_type in unique(data[Symbol(plotting_field)])
        filtered_data = data[data[Symbol(plotting_field)] .== field_type, :]
        value = filtered_data[:normalized_discounted_reward]
        push!(df, (field_type, mean(value), std(value)/sqrt(size(filtered_data,1))))
    end
    
    return plot(x=df.ParameterValue, y=df.MeanValue, ymin=(df.MeanValue - df.SEMValue), ymax=(df.MeanValue + df.SEMValue),
                color=df.ParameterValue, Geom.point, Geom.errorbar, 
                Guide.xlabel(xlabel), Guide.ylabel("Value"))
end

function plot_value_SEM(data::DataFrame, xfield, yfield)
    xtype = typeof(first(data[Symbol(xfield)]))
    ytype = typeof(first(data[Symbol(yfield)]))
    
    best_params = split(best_parameters(data), "_")
    solver_fields = solver_parameter_map[best_params[1]]
    for (i, field) in enumerate(solver_fields)
        field_type = typeof(first(data[Symbol(field)]))
        if !(field == xfield) && !(field == yfield)
            data = data[data[Symbol(field)] .== parse(field_type, best_params[i+1]), :] 
        end
    end

    for field1 in unique(data[Symbol(xfield)])
        for field2 in unique(data[Symbol(yfield)])
            filtered_data = data[(data[Symbol(xfield)] .== field1).&(data[Symbol(yfield)] .== field2), :]
            value = filtered_data[:normalized_discounted_reward]
            push!(df, (field1.*field2, mean(value), std(value)/sqrt(size(filtered_data,1))))
        end
    end
    
    return plot(x=df.ParameterValue, y=df.MeanValue, ymin=(df.MeanValue - df.SEMValue), ymax=(df.MeanValue + df.SEMValue),
                color=df.ParameterValue, Geom.point, Geom.errorbar, 
                Guide.xlabel(xlabel), Guide.ylabel("Value"))

end
