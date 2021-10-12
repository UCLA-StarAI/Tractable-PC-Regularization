# Run the experiments

- Generate Strudel models by running 

```
sh learn_strudel.sh
```

The resultant PCs should be saved in the folder "./pcs".

- Learn parameters with Laplace Smoothing:

```
julia run_all_exps1.jl
```

- Learn parameters with Soft Regularization:

```
julia run_all_exps2.jl
```

- Learn parameters with Entropy Regularization:

```
julia run_all_exps3.jl
```

- Learn parameters with Soft Regularization + Entropy Regularization:

```
julia run_all_exps5.jl
```

- Parse the results using "parse_results.ipynb"