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

recorded_pc_names = Vector{String}()
if isfile("entropy_reg_logfile-new3.txt")
    rm("entropy_reg_logfile-new3.txt")
end

ent_reg_candidates = [0.001, 0.01, 0.04, 0.1]
ent_reg_res = Vector{Float64}()
res = @distributed (+) for item in pc_and_dataset_names
    dataset_name, pc_name = item
    if pc_name in recorded_pc_names
        println("skip $(pc_name)")
    else
        pc, train_data, valid_data, test_data = get_pc_and_data(dataset_name, pc_name);
        if pc !== nothing
            best_ent_reg = 0.0
            best_result = (-Inf, -Inf, -Inf)
            for entropy_reg in ent_reg_candidates
                result = train_psdd(pc, train_data, valid_data, test_data; pseudocount = 2.0, entropy_reg);
                if result[2] > best_result[2]
                    best_ent_reg = entropy_reg
                    best_result = result
                end
            end
            print("dataset $(dataset_name), entropy reg $(best_ent_reg), ");
            println(best_result);
            push!(ent_reg_res, best_result[3]);
            open("entropy_reg_logfile-new3.txt", "a+") do io
                write(io, "$(dataset_name) $(pc_name) $(best_result[3])\n")
            end
        end
    end
    0
end