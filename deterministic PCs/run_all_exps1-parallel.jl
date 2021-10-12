using LogicCircuits
using ProbabilisticCircuits
using CUDA
using Plots
using Distributed

@everywhere include("./weighted_laplase_smoothing.jl")
@everywhere include("./train_scripts.jl")


function list_pc_and_dataset_names()
    pc_names = readdir("pcs")
    dataset_names = map(pc_names) do pc_name
        if length(split(pc_name, "_")) == 2
            split(pc_name, "_")[1]
        else
            split(pc_name, "_")[1] * "_" * split(pc_name, "_")[2]
        end
    end
    
    pc_names, dataset_names
end


@everywhere function get_pc_and_data(dataset_name, pc_name)
    train_data, valid_data, test_data = twenty_datasets(dataset_name)
    pc = load_prob_circuit("./pcs/" * pc_name)
    
    pc, train_data, valid_data, test_data
end


function filter_pcs(pc_names, dataset_names)
    new_pc_names = Vector{String}()
    new_dataset_names = Vector{String}()
    
    for (pc_name, dataset_name) in zip(pc_names, dataset_names)
        iters = parse(Int64, split(split(pc_name, "_")[end], ".")[1])
        if iters <= 4000
            push!(new_pc_names, pc_name)
            push!(new_dataset_names, dataset_name)
        end
    end
    
    new_pc_names, new_dataset_names
end


pc_names, dataset_names = list_pc_and_dataset_names()
pc_names, dataset_names = filter_pcs(pc_names, dataset_names)
println(length(pc_names))

pc_and_dataset_names = [(dataset_name, pc_name) for (dataset_name, pc_name) in zip(dataset_names, pc_names)]

best_pseudocount_res = Vector{Float64}()
degree_of_overfitting = Vector{Float64}()
res = @distributed (+) for item in pc_and_dataset_names
    pseudocount_candidates = [0.1, 0.4, 1.0, 2.0, 4.0, 10.0]
    dataset_name, pc_name = item
    pc, train_data, valid_data, test_data = get_pc_and_data(dataset_name, pc_name);
    if pc !== nothing
        best_pseudocount = 0.0
        best_result = (-Inf, -Inf, -Inf)
        for pseudocount in pseudocount_candidates
            result = train_psdd(pc, train_data, valid_data, test_data; pseudocount = 1.0);
            if result[2] > best_result[2]
                best_pseudocount = pseudocount
                best_result = result
            end
        end
        print("dataset $(dataset_name), ");
        println(best_result);
        open("pseudocount_logfile3.txt", "a+") do io
            write(io, "$(dataset_name) $(pc_name) $(best_result[3]) $((best_result[1] - best_result[2]) / abs(best_result[2]))\n")
        end
    end
    0
end