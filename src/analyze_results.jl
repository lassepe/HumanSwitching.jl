sem(v::AbstractVector) = std(v)/sqrt(length(v))


function get_result_plot_stack(data::DataFrame)
    Gadfly.set_default_plot_size(15cm,80cm)
    legend_guide = Guide.colorkey(title="Legend")
    xticks_guide = Guide.xticks(orientation=:horizontal)
    default_font = "cmr10"
    plot_stack = []
    default_theme = Gadfly.Theme(key_max_columns=2,
                                 plot_padding=[0mm],
                                 key_title_font=default_font,
                                 key_title_font_size=0mm,
                                 key_label_font=default_font,
                                 key_label_font_size=10pt,
                                 major_label_font=default_font,
                                 minor_label_font=default_font,
                                 major_label_font_size=10pt,
                                 minor_label_font_size=8pt, key_position=:top)

    # and some more for the remaining plots
    for i in 1:20
        Gadfly.push_theme(default_theme)
    end
    # the theme for the first plot
    first_plot_theme = copy(default_theme)
    first_plot_theme.key_position=:top
    Gadfly.push_theme(first_plot_theme)

    fst_theme = copy(first_plot_theme)
    fst_theme.major_label_font_size=8pt

    # compute relevant statistics on data
    data_stats = DataFrame(Model=String[], MeanValue=Float64[], SEMValue=Float64[], MeanCompute=Float64[], SEMCompute=Float64[], MeanNSteps=Float64[], SEMNSteps=Float64[])
	for planner_type in unique(data.planner_hbm_key)
		for solver_type in unique(data.solver_setup_key)
			common_rows = data[(data[:planner_hbm_key] .== planner_type).&(data[:solver_setup_key] .== solver_type), :]
			value = common_rows[:discounted_reward]
			compute = common_rows[:combined_median_time]
            n_steps = common_rows[:n_steps]
            N = size(common_rows,1)
			push!(data_stats, (solver_type,
                               mean(value), sem(value),
                               mean(compute), sem(compute),
                               mean(n_steps), sem(n_steps)
                              ))
		end
	end

    # Value vs. Compute (Scatter)
	value_v_compute_scatter = plot(data, x=:combined_median_time, y=:discounted_reward, color=:solver_setup_key, Geom.point,
                                   Guide.xlabel("CPU-Time per Decision [s]"), Guide.ylabel("Cumulative Discoutned Reward", orientation=:vertical), Gadfly.Scale.x_log10,
                                   legend_guide, xticks_guide)
    push!(plot_stack, value_v_compute_scatter)
    Gadfly.pop_theme()

    # Value - Cumulative Discoutned Reward:
    value_v_solver_sem = plot(x=data_stats.Model, y=data_stats.MeanValue,
                              ymin=(data_stats.MeanValue - data_stats.SEMValue), ymax=(data_stats.MeanValue + data_stats.SEMValue),
                              color=data_stats.Model, Geom.point, Geom.errorbar,
                              Guide.xlabel("Policy"), Guide.ylabel("Cumulative Discoutned Reward (SEM)", orientation=:vertical),
                              legend_guide, xticks_guide)
    push!(plot_stack, value_v_solver_sem)
    Gadfly.pop_theme()
    value_v_solver_density = plot(data, x=:discounted_reward, color=:solver_setup_key, Geom.density,
                                  Guide.xlabel("Cumulative Discoutned Reward"),
                                  legend_guide, xticks_guide)
    push!(plot_stack, value_v_solver_density)
    Gadfly.pop_theme()

    # Efficiency - NSteps (TODO: not fair because POMCPOW makes it more often. How to count N-Steps for policies that did not make it?):
    nstep_v_solver_sem = plot(x=data_stats.Model, y=data_stats.MeanNSteps,
                              ymin=(data_stats.MeanNSteps - data_stats.SEMNSteps), ymax=(data_stats.MeanNSteps + data_stats.SEMNSteps),
                              color=data_stats.Model, Geom.point, Geom.errorbar, Guide.xlabel("Policy"), Guide.ylabel("Number of Steps (SEM)", orientation=:vertical),
                              legend_guide, xticks_guide)
    push!(plot_stack, nstep_v_solver_sem)
    nstep_v_solver_histogram = plot(data, x=:n_steps, color=:solver_setup_key, Geom.histogram, Guide.xlabel("Number of Steps"),
                                    legend_guide, xticks_guide)
    push!(plot_stack, nstep_v_solver_histogram)

    # Compute
    compute_v_solver_sem = plot(x=data_stats.Model, y=data_stats.MeanCompute,
                                ymin=(data_stats.MeanCompute - data_stats.SEMCompute), ymax=(data_stats.MeanCompute + data_stats.SEMCompute),
                                color=data_stats.Model, Geom.point, Geom.errorbar,
                                Guide.xlabel("Policy"), Guide.ylabel("Cumulative Discoutned Reward (SEM)", orientation=:vertical),
                                legend_guide, xticks_guide)
    push!(plot_stack, compute_v_solver_sem)
    compute_v_solver_density = plot(data, x=:combined_median_time, color=:solver_setup_key, Geom.density,
                                    Guide.xlabel("CPU-Time per Decision [s]"),
                                    legend_guide, xticks_guide)
    push!(plot_stack, compute_v_solver_density)

    # Outcome (Success / Failure)
	outcome_histogram =  plot(data, xgroup=:solver_setup_key, x=:final_state_type, color=:solver_setup_key, Geom.subplot_grid(Geom.histogram),
                              fst_theme, Guide.xlabel("Outcome 𝐛𝐲 Policy"),
                              legend_guide)
    push!(plot_stack, outcome_histogram)

    return plot_stack
end
plot_results(args...) = display(vstack(get_result_plot_stack(args...)...))

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
		plot_results(data[data[:simulation_hbm_key] .== simulation_type, :])
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

                tail_expectation: $(tail_expectation(planner_data.discounted_reward, 0.20))
                mean: $(mean(planner_data.discounted_reward))
                lcb: $(mean(planner_data.discounted_reward) - std(planner_data.discounted_reward))
                """)
    end
end

# Rename:
#
# using CSV, DataFramesMeta
# rename(s::String, old::String="MostLikelyStateController", new::String="MLRA") = return s == old ? new : s;
# data = CSV.read(...)
#
# d = HS.transform_data(data)
# d = @transform(d, solver_setup_key=rename.(:solver_setup_key, "ProbObstacles", "PSRP"))
#
# CSV.write("path/to/file.csv", d)

function generate_eval_plots(data=nothing;
                             filename::String="$(@__DIR__)/../results/final_results/data_POMCPOW_PSRP.csv",
                             outdir::String="$(@__DIR__)/../results/final_results/plots/")

    if isnothing(data)
        data = CSV.read(filename)
    end

    plot_stack = get_result_plot_stack(data);
    (value_compute_scatter,
     value_sem, value_density,
     nstep_sem, nstep_density,
     compute_violin, compute_density,
     outcome_histogram) = plot_stack

    # value-compute Scatter
    dims = (14.5cm, 8cm)
    draw(PDF(joinpath(outdir, "hri_value_compute_scatter_plot.pdf"), dims...), value_compute_scatter)

    # value SEM
    dims = (14.5cm, 8cm)
    draw(PDF(joinpath(outdir, "hri_value_sem_plot.pdf"), dims...), value_sem)

    # value density
    dims = (14.5cm, 8cm)
    draw(PDF(joinpath(outdir, "hri_value_density_plot.pdf"), dims...), value_density)

    # nstep SEM
    dims = (14.5cm, 8cm)
    draw(PDF(joinpath(outdir, "hri_nstep_sem_plot.pdf"), dims...), nstep_sem)

    # nstep density
    dims = (14.5cm, 8cm)
    draw(PDF(joinpath(outdir, "hri_nstep_density_plot.pdf"), dims...), nstep_density)

    # compute density
    dims = (14.5cm, 8cm)
    draw(PDF(joinpath(outdir, "hri_compute_density_plot.pdf"), dims...), compute_density)

    # compute density
    dims = (14.5cm, 8cm)
    draw(PDF(joinpath(outdir, "hri_outcome_histogram_plot.pdf"), dims...), outcome_histogram)
end
