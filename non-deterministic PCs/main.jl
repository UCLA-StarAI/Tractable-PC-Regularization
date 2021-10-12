using LogicCircuits
using ProbabilisticCircuits
using Printf: @printf
using CUDA
using Random

include("structures.jl")


function main(dataset_id = 1, gpu = 1, seed = 1)
    Random.seed!(seed)
    
    # Whether to use gpu
    use_gpu = true
    
    # Specify CUDA device to use
    if use_gpu
        device!(collect(devices())[gpu])
    end
    
    # Mini-batch size
    batch_size = 1024
    
    num_hidden_cats = 12
    num_trees = 4
    
    pseudocount = 0.1
    soft_reg = 0.002
    entropy_reg = 0.001
    
    # Number of EM iterations
    minibatch_em_iters = 100
    fullbatch_em_iters = 100
    
    # Minibatch-update factor annealing
    exp_update_factor_start = 0.1
    exp_update_factor_end = 0.9
    
    # Maximum tolerance for validation accuracy increase
    max_tol_iters = 3
    
    print("Loading dataset $(twenty_dataset_names[dataset_id])")
    t = @elapsed begin
        train_data, valid_data, test_data = twenty_datasets(twenty_dataset_names[dataset_id])
        
        num_vars = num_features(train_data)
    end
    @printf(" (%.3fs)\n", t)
    println("Features: $(num_features(train_data)); Train Examples: $(num_examples(train_data)); Test Examples: $(num_examples(test_data))")

    print("Generating HCLT circuit")
    t = @elapsed begin
        circuit = hidden_chow_liu_circuit(num_vars; data = train_data, num_hidden_cats, 
                                          num_trees, num_tree_candidates = 20)
        uniform_parameters(circuit; perturbation = 0.4)
    end
    @printf(" (%.3fs)\n", t)
    println("Number of edges: $(num_edges(circuit)); Number of parameters: $(num_parameters(circuit))")
    
    folder_name = "logs_$(num_hidden_cats)_$(num_trees)_$(pseudocount)_$(soft_reg)_$(entropy_reg)_$(seed)"
    mkpath(folder_name)
    file_name = "$(twenty_dataset_names[dataset_id])"
    open("$(folder_name)/$(file_name).txt", "w") do io
        write(io, "")
    end
    
    # Apply soft regularization
    train_data = soften(train_data, soft_reg; scale_by_marginal = true)

    # Mini-batch EM steps
    last_valid_ll = typemin(Float64)
    ll_tol_iter = 0
    for iter = 1 : minibatch_em_iters

        @printf("      - Mini-batch EM iter %d:\n", iter)

        # To estimate circuit parameters under the existence of latent/hidden variables, we employ the 
        # Expectation-Maximization (EM) algorithm.
        print("        - Estimating parameters with EM")
        t = @elapsed begin
            exp_update_factor = exp_update_factor_start + (iter - 1) * (exp_update_factor_end - exp_update_factor_start) / (minibatch_em_iters - 1)
            estimate_parameters_em(circuit, batch(train_data, batch_size); pseudocount, entropy_reg, use_gpu, 
                                   exp_update_factor, update_per_batch = true)
        end
        @printf(" (%.3fs)\n", t)

        # Evaluate the test set likelihood 
        print("        - Evaluating marginal log-likelihoods")
        t = @elapsed begin
            train_ll = marginal_log_likelihood_avg(circuit, batch(train_data, batch_size); use_gpu)
            valid_ll = marginal_log_likelihood_avg(circuit, batch(valid_data, batch_size); use_gpu)
            test_ll = marginal_log_likelihood_avg(circuit, batch(test_data, batch_size); use_gpu)
        end
        @printf(" (%.3fs)\n", t)
        @printf("        - Train: \033[1m%.3f\033[0m; Valid: \033[1m%.3f\033[0m; Test: \033[1m%.3f\033[0m\n", 
                train_ll, valid_ll, test_ll)
        
        open("$(folder_name)/$(file_name).txt", "a+") do io
            write(io, "$(train_ll) $(valid_ll) $(test_ll)\n")
        end
        
        if valid_ll < last_valid_ll
            ll_tol_iter += 1
            if ll_tol_iter > max_tol_iters
                break
            end
        else
            ll_tol_iter = 0
        end
        last_valid_ll = valid_ll
    end
    
    # Full batch EM steps
    last_valid_ll = typemin(Float64)
    ll_tol_iter = 0
    for iter = 1 : fullbatch_em_iters

        @printf("      - Full-batch EM iter %d:\n", iter)

        # To estimate circuit parameters under the existence of latent/hidden variables, we employ the 
        # Expectation-Maximization (EM) algorithm.
        print("        - Estimating parameters with EM")
        t = @elapsed begin
            estimate_parameters_em(circuit, batch(train_data, batch_size); pseudocount, entropy_reg, use_gpu)
        end
        @printf(" (%.3fs)\n", t)

        # Evaluate the test set likelihood 
        print("        - Evaluating marginal log-likelihoods")
        t = @elapsed begin
            train_ll = marginal_log_likelihood_avg(circuit, batch(train_data, batch_size); use_gpu)
            valid_ll = marginal_log_likelihood_avg(circuit, batch(valid_data, batch_size); use_gpu)
            test_ll = marginal_log_likelihood_avg(circuit, batch(test_data, batch_size); use_gpu)
        end
        @printf(" (%.3fs)\n", t)
        @printf("        - Train: \033[1m%.3f\033[0m; Valid: \033[1m%.3f\033[0m; Test: \033[1m%.3f\033[0m\n", 
                train_ll, valid_ll, test_ll)
        
        open("$(folder_name)/$(file_name).txt", "a+") do io
            write(io, "$(train_ll) $(valid_ll) $(test_ll)\n")
        end
        
        if valid_ll < last_valid_ll
            ll_tol_iter += 1
            if ll_tol_iter > max_tol_iters
                break
            end
        else
            ll_tol_iter = 0
        end
        last_valid_ll = valid_ll
    end
    
end


dataset_id = length(ARGS) >= 2 ? parse(Int64, ARGS[1]) : 1
gpu = length(ARGS) >= 2 ? parse(Int64, ARGS[2]) : 1
seed = length(ARGS) >= 3 ? parse(Int64, ARGS[3]) : 1
main(dataset_id, gpu, seed)