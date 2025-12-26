# === AdaConditionalTransformer utilities ===

"""
    default_pos_mask(tokens; start_token_id)

Return a mask of shape (1, seq_len, batch) that is 0 before the first
occurrence of `start_token_id` (usually '>') and 1 from that position onwards.

This is exactly the "only condition after `>`" behaviour we want for both
training and generation.
"""
function default_pos_mask(tokens::AbstractArray{<:Integer}; start_token_id::Integer)
    is_start      = tokens .== start_token_id              # (seq, batch)
    cumsum_start  = cumsum(is_start, dims = 1)             # (seq, batch)
    m             = Float32.(cumsum_start .> 0)            # 0/1
    return reshape(m, 1, size(tokens, 1), size(tokens, 2)) # (1, seq, batch)
end

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

function AdaConditionalTransformer(
    cond_embeddings::Tuple,
    vocab_size::Int, dim::Int, n_layers::Int, n_heads::Int,
    n_kv_heads::Int, max_seq_len::Int, ff_hidden_dim::Int;
    norm_eps = 1f-5,
    rope_settings = (theta = 500000f0, use_scaled = false, scale_factor = 8),
    head_dim = dim ÷ n_heads,
    kws...
)
    tok_embeddings = Embedding(vocab_size => dim)

    layers = Tuple(
        AdaTransformerBlock(dim, n_heads, n_kv_heads, ff_hidden_dim;
                            norm_eps, head_dim, kws...)
        for _ in 1:n_layers
    )

    norm   = RMSNorm(dim, eps = norm_eps)
    output = Dense(dim => vocab_size, bias = false)
    rope   = RoPE(head_dim, max_seq_len * 2; rope_settings...)

    AdaConditionalTransformer(tok_embeddings, cond_embeddings, layers, norm, output, rope)
end

# --- cache helpers (same pattern as ConditionalTransformer) ---

function no_kv_cache(model::AdaConditionalTransformer)
    return Tuple(no_kv_cache(layer.attention) for layer in model.layers)
end

function kv_cache(model::AdaConditionalTransformer, seq_length::Int, batch_size::Int = 1)
    return Tuple(kv_cache(layer.attention, seq_length, batch_size) for layer in model.layers)
end

# --- forward pass ---

function (model::AdaConditionalTransformer)(
    tokens::AbstractArray{Int},
    conditionals::Tuple;
    conditional_list     = 1:length(conditionals),
    conditional_mask_gen = default_conditional_mask,  # currently unused for Ada
    caches               = no_kv_cache(model),
    pos_mask             = nothing,
    start_token_id       = 2,                         # assumes '>' is token 2 in "<>"
    kws...
)
    # embeddings: (dim, seq, batch)
    h = model.tok_embeddings(tokens)

    # Sum all conditioning embeddings into a single (dim, batch) vector
    first_idx = first(conditional_list)
    cond = model.cond_embeddings[first_idx](conditionals[first_idx])
    for idx in Iterators.drop(conditional_list, 1)
        cond .+= model.cond_embeddings[idx](conditionals[idx])
    end

    # If caller didn't give a pos_mask, derive from tokens ("after >" behaviour)
    if pos_mask === nothing
        pos_mask = default_pos_mask(tokens; start_token_id = start_token_id)
    end

    # RoPE slice based on current cache position
    rope = model.rope[position(caches) .+ (1:size(tokens, 1))]

    # Each AdaTransformerBlock takes (h, cond, pos_mask; rope, cache, kws...)
    for (layer, cache) in zip(model.layers, caches)
        h = layer(h, cond, pos_mask; rope, cache, kws...)
    end

    h = model.norm(h)
    return model.output(h)
end

# --- loss wrapper (Flux-style) ---

function forward_loss(
    model::AdaConditionalTransformer,
    inputs::AbstractArray,
    conditionals::Tuple,
    targets::AbstractArray;
    loss_mask = nothing,
    kws...
)
    logits = model(inputs, conditionals; kws...)
    return loss(logits, targets, loss_mask = loss_mask)
end

# --- autoregressive generation ---

function generate(
    model::AdaConditionalTransformer,
    initial_tokens::AbstractArray{<:Integer},  # (seq_len,) or (seq_len,1)
    conditionals;
    io               = stdout,
    max_new_tokens   = 100,
    sampler::Function = argmax_sampler,
    tokenizer_for_printing = nothing,
    end_token        = 128010,
    caches           = kv_cache(model, 1024, 1),
    device           = identity,
    start_token_id   = 2,   # '>' token
    kws...
)
    # Represent tokens as (seq_len, 1)
    tokens = reshape(initial_tokens, :, 1)
    conditionals = device(conditionals)

    # ---- 1) Prefill with prefix (if any) ----
    n = size(tokens, 1)
    if n > 1
        prefix = tokens[1:n-1, :]

        pos_mask_prefix = default_pos_mask(prefix; start_token_id = start_token_id)
        pos_mask_prefix = device(pos_mask_prefix)

        model(
            device(prefix),
            conditionals;
            caches,
            mask     = causal_mask,
            pos_mask = pos_mask_prefix,
            kws...
        )
    end

    # ---- 2) Autoregressive generation ----
    for step in 1:max_new_tokens
        # Recompute "after >" mask on the fly for the whole sequence,
        # then take the last position only
        full_mask     = default_pos_mask(tokens; start_token_id = start_token_id)
        pos_mask_step = device(full_mask[:, end:end, :])  # (1,1,1)

        last_tok = device(tokens[end:end, :])             # (1,1)
        logits = model(
            last_tok,
            conditionals;
            caches,
            mask     = causal_mask,
            pos_mask = pos_mask_step,
            kws...
        )

        new_token = sampler(logits[:, end])
        tokens = vcat(tokens, reshape([new_token], 1, 1))

        if !isnothing(tokenizer_for_printing)
            print(
                io,
                decode(tokenizer_for_printing, [new_token] |> cpu;
                       skip_special_tokens = false),
            )
        end

        new_token == end_token && break
    end

    return tokens
end
