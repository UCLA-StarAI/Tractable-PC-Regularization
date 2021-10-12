using LogicCircuits
using ProbabilisticCircuits

# include("weighted_laplase_smoothing.jl")


function train_psdd(pc, train_data, valid_data, test_data; pseudocount = 1.0, soft_reg = 0.0, 
                    entropy_reg = 0.0, alpha = 0.0, use_gpu = false, batch_size = 1024)
    if pc === nothing
        return nothing, nothing, nothing
    end
    
    if soft_reg > 1e-8
        data = batch(soften(train_data, soft_reg; scale_by_marginal = true), batch_size)
    else
        data = batch(train_data, batch_size)
    end
    
    if alpha == 0.0
        estimate_parameters(pc, data; use_gpu, pseudocount, entropy_reg)
    else
        weighted_laplase_smoothing(pc, data; alpha)
    end
    
    train_ll = log_likelihood_avg(pc, batch(train_data, batch_size); use_gpu)
    valid_ll = log_likelihood_avg(pc, batch(valid_data, batch_size); use_gpu)
    test_ll = log_likelihood_avg(pc, batch(test_data, batch_size); use_gpu)
    
    train_ll, valid_ll, test_ll
end