
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