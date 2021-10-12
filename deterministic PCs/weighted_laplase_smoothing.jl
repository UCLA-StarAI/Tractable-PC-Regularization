using LogicCircuits
using ProbabilisticCircuits
using StatsFuns: logsumexp


MCCache = Dict{ProbCircuit, Float64}


function log_model_count(pc::ProbCircuit; cache = MCCache())::Float64
    f_con(n)::Float64 = error("`model_counts` does not support constant nodes.")
    f_lit(n)::Float64 = get!(cache, n, 0.0)
    f_a(n, cn)::Float64 = get!(cache, n, reduce(+, [cn...]))
    f_o(n, cn)::Float64 = get!(cache, n, logsumexp([cn...]))
    
    foldup_aggregate(pc, f_con, f_lit, f_a, f_o, Float64)
end


function weighted_laplase_smoothing(pc::ProbCircuit, data; alpha::Float64)
    cache = MCCache()
    log_model_count(pc; cache)
    
    bc = BitCircuit(pc, data)
    
    edge_counts::Vector{Float64} = zeros(Float64, num_elements(bc))
    parent_node_counts::Vector{Float64} = zeros(Float64, num_elements(bc))
    
    if isbinarydata(data)
        node_counts = Vector{UInt}(undef, num_nodes(bc))
    else
        node_counts = Vector{Float64}(undef, num_nodes(bc))
    end
    
    @inline function on_node(flows, values, dec_id, weights::Nothing)
        node_counts[dec_id] = sum(1:size(flows,1)) do i
            count_ones(flows[i, dec_id]) 
        end
    end
    @inline function on_node(flows, values, dec_id, weights)
        node_counts[dec_id] = sum(1:size(flows,1)) do i
            weighted_count_ones(flows[i, dec_id], i * 64 - 63, min(i * 64, length(weights)), weights)
        end
    end
    @inline function on_node(flows, values::Matrix{<:AbstractFloat}, dec_id, weights::Nothing)
        node_counts[dec_id] = sum(1:size(flows,1)) do i
            flows[i, dec_id]
        end
    end
    @inline function on_node(flows, values::Matrix{<:AbstractFloat}, dec_id, weights)
        node_counts[dec_id] = sum(1:size(flows,1)) do i
            flows[i, dec_id] * weights[i]
        end
    end

    @inline function estimate(element, decision, edge_count)
        edge_counts[element] += edge_count
        parent_node_counts[element] += node_counts[decision]
    end

    @inline function on_edge(flows, values, prime, sub, element, grandpa, single_child, weights::Nothing)
        if !single_child
            edge_count = sum(1:size(flows,1)) do i
                count_ones(values[i, prime] & values[i, sub] & flows[i, grandpa]) 
            end
            estimate(element, grandpa, edge_count)
        end # no need to estimate single child params, they are always prob 1
    end
    @inline function on_edge(flows, values::Matrix{<:AbstractFloat}, prime, sub, element, grandpa, single_child, weights::Nothing)
        if !single_child
            edge_count = sum(1:size(flows,1)) do i
                values[i, prime] * values[i, sub] / values[i, grandpa] * flows[i, grandpa]
            end
            estimate(element, grandpa, edge_count)
        end # no need to estimate single child params, they are always prob 1
    end
    
    if isbatched(data)
        v, f = nothing, nothing
        map(data) do d
            v, f = satisfies_flows(bc, d, v, f; on_node = on_node, on_edge = on_edge, weights = nothing)
        end
    else
        v, f = satisfies_flows(bc, data; on_node = on_node, on_edge = on_edge, weights = nothing)
    end
    
    foreach(pc) do pn
        if is⋁gate(pn)
            if num_children(pn) == 1
                pn.log_probs .= zero(Float64)
            else
                id = (bc.node2id[pn]::⋁NodeIds).node_id
                @inbounds els_start = bc.nodes[1,id]
                @inbounds els_end = bc.nodes[2,id]
                @inbounds log_model_counts = [cache[c] for c in children(pn)]
                @inbounds @views log_model_counts .-= logsumexp(log_model_counts)
                @inbounds @views pn.log_probs .= log.((edge_counts[els_start:els_end] .+ (alpha .* exp.(log_model_counts))) ./ (sum(edge_counts[els_start:els_end]) + alpha))
                if isnan(pn.log_probs[1])
                    println("edge counts: ", edge_counts[els_start:els_end])
                    println("model counts: ", model_counts)
                    println("alpha: ", alpha)
                end
                @assert isapprox(sum(exp.(pn.log_probs)), 1.0, atol=1e-3) "Parameters do not sum to one locally: $(sum(exp.(pn.log_probs))); $(pn.log_probs)"
                pn.log_probs .-= logsumexp(pn.log_probs) # normalize away any leftover error
            end
        end
    end
    
    nothing
end