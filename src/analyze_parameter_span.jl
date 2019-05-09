solver_parameter_map = Dict{String, Array{String}}(
                                                   "POMCP" => ["Depth", "UCB_criterion", "Time"],
                                                   "POMCPOW" => ["Depth", "UCB_criterion", "Time"],
                                                   "DESPOT" => ["Depth", "Scenarios", "Lambda", "Time"]
                                                  )

function process_data_span(data::DataFrame)
    function get_solver_params(solver_name::String)
        solver_pars = [parse(Float64, x) for x in split(solver_name, "_")[2:end]]
    end
    
    data_solvers = []
    for solver in keys(solver_parameter_map)
        solver_data = data[occursin.("$(solver)_", data[:solver_setup_key]), :]
        for (i,field) in enumerate(solver_parameter_map[solver])
            solver_data[Symbol(field)] = getindex.(get_solver_params.(solver_data[:solver_setup_key]), i)
        end
        push!(data_solvers, solver_data)
    end
    return data_solvers
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

function plot_param_comparison(data::DataFrame; static_fields::Bool=false)
    # Split data per solver
    data_per_solver = process_data_span(data)

    # Plot within-solver parameter comparison
    # Per solver Value vs Field plots
    for solver_data in data_per_solver
        plot_solver(solver_data, static_fields=static_fields)
    end

    # Plot overall solver parameter comparison
    # One Value vs Field plot with all solvers on the same plot
    # The Field must exist for all solvers
    for field in ["Time", "Depth"]
        plot_all_solvers(data_per_solver, field; static_fields=static_fields)
    end
end

function plot_all_solvers(data_per_solver, plotting_field::String; static_fields::Bool=false)
    Gadfly.set_default_plot_size(30cm,30cm)
	detailed_theme = Gadfly.Theme(minor_label_font_size=8pt, key_position=:none)
    
    plots = []
    for field in ["Time", "Depth"]
        push!(plots, plot_all_solver_field(data_per_solver, field, static_fields=static_fields))
    end
    
    final_plot = Gadfly.title(vstack(plots...), 
                             """
                             Problem Instance: $(first(first(data_per_solver)[:pi_key]))
                             True Human Model: $(first(first(data_per_solver)[:simulation_hbm_key]))
                             Planner Model: $(first(first(data_per_solver)[:planner_hbm_key]))
                             Solver Comparison
                             """)
    display(final_plot)
end

function plot_all_solver_field(data_per_solver, plotting_field::String; static_fields::Bool=false)
    df = DataFrame()
    for solver_data in data_per_solver
        solver_type = first(split(first(solver_data[:solver_setup_key]), "_"))
        solver_df = create_value_SEM(solver_data, plotting_field)
        solver_df[:Solver] = solver_type
        df = [df; solver_df]
    end
    return plot(x=df.ParameterValue, y=df.MeanValue, ymin=(df.MeanValue - df.SEMValue), ymax=(df.MeanValue + df.SEMValue),
                color=df.Solver, Geom.point, Geom.errorbar, 
                Guide.xlabel(plotting_field), Guide.ylabel("Value"))
end

function plot_solver(data::DataFrame; static_fields::Bool=false)
    Gadfly.set_default_plot_size(30cm,30cm)
	detailed_theme = Gadfly.Theme(minor_label_font_size=8pt, key_position=:none)

    solver_type = first(split(first(data[:solver_setup_key]), "_"))
    plots = []
    for field in solver_parameter_map[solver_type]
        push!(plots, plot_solver_field(data, field, static_fields=static_fields))
    end

    final_plot = Gadfly.title(hstack(plots...), 
                             """
                        	 Problem Instance: $(first(data[:pi_key]))
                         	 True Human Model: $(first(data[:simulation_hbm_key]))
                             Planner Model: $(first(data[:planner_hbm_key]))
                             Solver: $(solver_type)
                             """)
    display(final_plot)
end

function plot_solver_field(data::DataFrame, plotting_field::String; static_fields::Bool=false)
    df = create_value_SEM(data, plotting_field, static_fields=static_fields)
    return plot(x=df.ParameterValue, y=df.MeanValue, ymin=(df.MeanValue - df.SEMValue), ymax=(df.MeanValue + df.SEMValue),
                Geom.point, Geom.errorbar, Guide.xlabel(plotting_field), Guide.ylabel("Value"))
end

function create_value_SEM(data::DataFrame, plotting_field::String; static_fields::Bool=false)
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
    return df
end
    
