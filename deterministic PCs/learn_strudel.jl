using LogicCircuits, ProbabilisticCircuits


function main(dataset_name)
    train_data, valid_data, test_data = twenty_datasets(dataset_name)
    
    pc, vtree = learn_chow_liu_tree_circuit(train_data)
    
    pc, vtree = learn_circuit(train_data, pc, vtree; maxiter = 1000, return_vtree = true)
    save_as_psdd("./pcs/$(dataset_name)_1000.psdd", pc, vtree)
    
    total_iters = 1000
    for idx = 1 : 20
        pc, vtree = learn_circuit(train_data, pc, vtree; maxiter = 200, return_vtree = true)
        total_iters += 200
        
        save_as_psdd("./pcs/$(dataset_name)_$(string(total_iters)).psdd", pc, vtree)
    end
end

function main_dna(dataset_name = "dna")
    train_data, valid_data, test_data = twenty_datasets(dataset_name)
    
    pc, vtree = learn_chow_liu_tree_circuit(train_data)
    
    total_iters = 0
    for idx = 1 : 20
        pc, vtree = learn_circuit(train_data, pc, vtree; maxiter = 50, return_vtree = true)
        total_iters += 50
        
        save_as_psdd("./pcs/$(dataset_name)_$(string(total_iters)).psdd", pc, vtree)
    end
end

# dataset_name = ARGS[1]
# main(dataset_name)

main_dna()