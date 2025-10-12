
# === AdaConditionalTransformer ===
@concrete struct AdaConditionalTransformer
    tok_embeddings
    cond_embeddings
    layers
    norm
    output
    rope
end

Flux.@layer AdaConditionalTransformer

function AdaConditionalTransformer(cond_embeddings::Tuple,
    vocab_size::Int, dim::Int, n_layers::Int, n_heads::Int, 
    n_kv_heads::Int, max_seq_len::Int, ff_hidden_dim::Int;
    norm_eps=1f-5,
    rope_settings = (theta = 500000f0, use_scaled = false, scale_factor = 8),
    head_dim = dim ÷ n_heads,
    kws...
)
    tok_embeddings = Embedding(vocab_size => dim)
    layers = Tuple(AdaTransformerBlock(dim, n_heads, n_kv_heads, ff_hidden_dim; norm_eps, head_dim, kws...) for _ in 1:n_layers)
    norm = RMSNorm(dim, eps=norm_eps)
    output = Dense(dim => vocab_size, bias=false)
    rope = RoPE(head_dim, max_seq_len * 2; rope_settings...)
    AdaConditionalTransformer(tok_embeddings, cond_embeddings, layers, norm, output, rope)
end

function (model::AdaConditionalTransformer)(tokens::AbstractArray{Int}, conditionals::Tuple; 
                                            conditional_list = 1:length(conditionals), 
                                            conditional_mask_gen = default_conditional_mask, 
                                            caches = no_kv_cache(model), kws...)
    h = model.tok_embeddings(tokens) # (dim, seq_len, batch)
    
    # Sum all conditioning embeddings into one global cond vector
    cond = model.cond_embeddings[first(conditional_list)](conditionals[first(conditional_list)])
    for (ic, c) in enumerate(conditional_list)
        ic == 1 && continue
        cond = cond .+ model.cond_embeddings[c](conditionals[ic])
    end
    # cond is now (dim, batch)
    
    rope = model.rope[position(caches) .+ (1:size(tokens, 1))]
    for (layer, cache) in zip(model.layers, caches)
        h = layer(h, cond; rope, cache, kws...)  # Pass cond to block!
    end
    h = model.norm(h)
    output = model.output(h)
    return output
end

forward_loss(model::AdaConditionalTransformer, inputs::AbstractArray, conditionals, targets::AbstractArray; loss_mask = nothing, kws...) = 
    loss(model(inputs, conditionals; kws...), targets, loss_mask = loss_mask)

function generate(
    model::AdaConditionalTransformer, 
    initial_tokens::AbstractArray{<:Integer},
    conditionals;
    io=stdout,
    max_new_tokens=100,
    sampler::Function=argmax_sampler,
    tokenizer_for_printing = nothing,
    end_token = 128010,
    caches=kv_cache(model, 1024, 1),
    device=identity,
    kws...
)
    n, b = size(initial_tokens, 1), size(initial_tokens, 2)
    tokens = reshape(initial_tokens, n, b)
    conditionals = device(conditionals)
    
    # Process initial tokens if n > 1
    if n > 1
        # Create pos_mask for initial sequence
        start_tok_id = 2  # Assuming '>' is token 2 from your AAs alphabet
        start_pos = findfirst(==(start_tok_id), initial_tokens)
        
        # 0 before '>', 1 at and after '>'
        pos_mask = Float32.(reshape(1:n-1, 1, n-1, 1) .>= (start_pos === nothing ? n+1 : start_pos))
        pos_mask = device(pos_mask)
        
        model(device(tokens[1:n-1, :]), conditionals; caches, mask=causal_mask, pos_mask, kws...)
    end
    
    # Generate new tokens - all get full conditioning (pos_mask=1)
    pos_mask_single = device(ones(Float32, 1, 1, 1))
    
    for i in 1:max_new_tokens
        logits = model(device(reshape(tokens[end, :], 1, b)), conditionals; caches, pos_mask=pos_mask_single, kws...)
        new_token = sampler(logits[:, end])
        tokens = vcat(tokens, reshape([new_token], 1, b))
        !isnothing(tokenizer_for_printing) && print(io, decode(tokenizer_for_printing, [new_token] |> cpu, skip_special_tokens = false))
        new_token == end_token && break
    end
    
    return tokens
end