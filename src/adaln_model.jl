# ============================================================
#  Simple AdaLN Conditional Transformer
#    - No gates
#    - No cond_mlp
#    - Uses ConditionalMask-style conditional_mask_gen
#    - Sampling: prefill WITH causal mask, decode WITHOUT
# ============================================================

# Assumes these exist in Jjama3 scope:
#   Attention, FeedForward, AdaLN, RMSNorm, RoPE
#   causal_mask, loss, argmax_sampler, decode
#   kv_cache, no_kv_cache, position (from cache.jl)
#
# And your additive model provides ConditionalMask(1) which matches:
#   conditional_mask_gen(tokens, conditional_list) -> (seq, batch, ncond)

# ------------------------------------------------------------
# default_pos_mask: fallback if no ConditionalMask is provided
# ------------------------------------------------------------
function default_pos_mask(tokens::AbstractArray{<:Integer}; start_token_id::Integer)
    is_start      = tokens .== start_token_id              # (seq, batch)
    cumsum_start  = cumsum(is_start, dims = 1)             # (seq, batch)
    m             = Float32.(cumsum_start .> 0)            # 0/1
    return reshape(m, 1, size(tokens, 1), size(tokens, 2)) # (1, seq, batch)
end

# --- helper: (seq,batch,ncond) -> (1,seq,batch) ---
conditional_mask_to_pos_mask(cm) = begin
    pm = maximum(cm; dims = 3)                       # (seq, batch, 1)
    pm = dropdims(pm; dims = 3)                      # (seq, batch)
    reshape(Float32.(pm), 1, size(pm,1), size(pm,2)) # (1, seq, batch)
end


# ------------------------------------------------------------
# AdaTransformerBlock: AdaLN + Attention/FFN, no gates
# ------------------------------------------------------------
@concrete struct AdaTransformerBlock
    attention
    feed_forward
    attention_adaln
    ffn_adaln
end

Flux.@layer AdaTransformerBlock

function AdaTransformerBlock(
    in_dim::Int, n_heads::Int, n_kv_heads::Int = n_heads, ff_hidden_dim::Int = 4 * in_dim;
    norm_eps = 1f-5, head_dim = in_dim ÷ n_heads, kws...
)
    AdaTransformerBlock(
        Attention(in_dim, n_heads, n_kv_heads; head_dim, kws...),
        FeedForward(in_dim, ff_hidden_dim),
        AdaLN(in_dim, in_dim; norm_eps = norm_eps),
        AdaLN(in_dim, in_dim; norm_eps = norm_eps),
    )
end

function (block::AdaTransformerBlock)(x, cond, pos_mask = nothing; kws...)
    # Attention
    x_mod = block.attention_adaln(x, cond, pos_mask)
    h     = x .+ block.attention(x_mod; kws...)

    # FFN
    h_mod = block.ffn_adaln(h, cond, pos_mask)
    out   = h .+ block.feed_forward(h_mod)

    return out
end


# ------------------------------------------------------------
# AdaConditionalTransformer
#   - sums conditional embeddings (no MLP)
#   - injects via AdaLN with pos_mask
# ------------------------------------------------------------
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
                              norm_eps = norm_eps, head_dim = head_dim, kws...)
        for _ in 1:n_layers
    )

    norm   = RMSNorm(dim, eps = norm_eps)
    output = Dense(dim => vocab_size, bias = false)
    rope   = RoPE(head_dim, max_seq_len * 2; rope_settings...)

    AdaConditionalTransformer(tok_embeddings, cond_embeddings, layers, norm, output, rope)
end


# ------------------------------------------------------------
# Forward
#   - conditional_list semantics match ConditionalTransformer:
#       ic indexes into `conditionals`
#       c  indexes into `cond_embeddings`
# ------------------------------------------------------------
function (model::AdaConditionalTransformer)(
    tokens::AbstractArray{<:Integer},
    conditionals::Tuple;
    conditional_list     = 1:length(conditionals),
    conditional_mask_gen = nothing,             # pass ConditionalMask(1) here
    caches               = no_kv_cache(model),
    pos_mask             = nothing,
    start_token_id       = 2,                   # assumes '>' is token 2 in "<>ACDE..."
    kws...
)
    h = model.tok_embeddings(tokens)  # (dim, seq, batch)

    # ---- Sum conditioning channels (NO MLP) ----
    cond = nothing
    for (ic, c) in enumerate(conditional_list)
        cond_i = model.cond_embeddings[c](conditionals[ic])  # (dim, batch)
        cond = (cond === nothing) ? cond_i : (cond .+ cond_i)
    end
    @assert cond !== nothing "AdaConditionalTransformer expects at least one conditional"

    # ---- Build pos_mask from ConditionalMask if given ----
    if pos_mask === nothing
        if conditional_mask_gen === nothing
            pos_mask = default_pos_mask(tokens; start_token_id = start_token_id)   # (1, seq, batch)
        else
            cm = Flux.ChainRulesCore.ignore_derivatives() do
                conditional_mask_gen(tokens, conditional_list)  # (seq, batch, ncond)
            end
            pos_mask = conditional_mask_to_pos_mask(cm)         # (1, seq, batch)
        end
    end

    rope = model.rope[position(caches) .+ (1:size(tokens, 1))]

    for (layer, cache) in zip(model.layers, caches)
        h = layer(h, cond, pos_mask; rope, cache, kws...)
    end

    h = model.norm(h)
    return model.output(h)
end


# ------------------------------------------------------------
# Loss wrapper (same style as Jjama3.forward_loss)
# ------------------------------------------------------------
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


# ------------------------------------------------------------
# Sampling (identical style to ConditionalTransformer)
#   Prefill WITH causal mask
#   Decode WITHOUT causal mask
# ------------------------------------------------------------
function generate(
    model::AdaConditionalTransformer,
    initial_tokens::AbstractArray{<:Integer},
    conditionals;
    io = stdout,
    max_new_tokens = 100,
    sampler::Function = argmax_sampler,
    tokenizer_for_printing = nothing,
    end_token = 128010,
    caches = kv_cache(model, 1024, 1),
    device = identity,
    conditional_mask_gen = nothing,   # pass ConditionalMask(1) here
    kws...
)
    # Make (seq, batch)
    if ndims(initial_tokens) == 1
        tokens = reshape(initial_tokens, :, 1)
    else
        n, b = size(initial_tokens, 1), size(initial_tokens, 2)
        tokens = reshape(initial_tokens, n, b)
    end

    n, b = size(tokens, 1), size(tokens, 2)
    conditionals = device(conditionals)

    # Prefill (WITH causal mask)
    n > 1 && model(
        device(tokens[1:n-1, :]),
        conditionals;
        caches,
        mask = causal_mask,
        conditional_mask_gen = conditional_mask_gen,
        kws...
    )

    # Decode (NO causal mask)
    for i in 1:max_new_tokens
        logits = model(
            device(tokens[end:end, :]),
            conditionals;
            caches,
            conditional_mask_gen = conditional_mask_gen,
            kws...
        )

        new_token = sampler(logits[:, end])
        tokens = [tokens; new_token]

        !isnothing(tokenizer_for_printing) && print(
            io,
            decode(tokenizer_for_printing, tokens[end:end] |> cpu; skip_special_tokens = false),
        )

        sum(tokens[end:end]) == end_token && break
    end

    return tokens
end