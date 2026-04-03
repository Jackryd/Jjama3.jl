
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
    cond_dim::Int = dim,
    kws...
)
    tok_embeddings = Embedding(vocab_size => dim)

    layers = Tuple(
        AdaTransformerBlock(dim, n_heads, n_kv_heads, ff_hidden_dim;
                              norm_eps = norm_eps, head_dim = head_dim, cond_dim = cond_dim, kws...)
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
    conditional_mask_gen = default_conditional_mask,
    caches               = no_kv_cache(model),
    kws...
)
    # --- embeddings ---
    h = model.tok_embeddings(tokens)
    if ndims(h) == 2
        h = reshape(h, size(h,1), size(h,2), 1)  # (dim, seq, 1)
    end

    # --- handle empty conditionals (eq-model in tree code) ---
    if isempty(conditional_list)
        cond = similar(h, size(h,1), size(h,3)); cond .= 0
        pos_mask = similar(h, Float32, 1, size(h,2), size(h,3)); pos_mask .= 0
    else
        # --- same mask API as ConditionalTransformer ---
        cm = Flux.ChainRulesCore.ignore_derivatives() do
            conditional_mask_gen(tokens, conditional_list)  # (seq,batch,ncond)
        end
        pm = maximum(cm; dims=3) |> x -> dropdims(x; dims=3)            # (seq,batch)
        pos_mask = reshape(Float32.(pm), 1, size(pm,1), size(pm,2))     # (1,seq,batch)

        # --- sum conditioning channels ---
        cond = nothing
        for (ic, c) in enumerate(conditional_list)
            ci = model.cond_embeddings[c](conditionals[ic])  # (dim,batch)
            cond = (cond === nothing) ? ci : (cond .+ ci)
        end
    end

    rope = model.rope[position(caches) .+ (1:size(tokens, 1))]
    checkpoint_layers = get(kws, :checkpoint_layers, false)
    kw_nt = (; kws...)
    layer_kws = Base.structdiff(kw_nt, (checkpoint_layers = checkpoint_layers,))

    for (layer, cache) in zip(model.layers, caches)
        if checkpoint_layers
            h = Flux.Zygote.checkpointed(
                (h_, c_, pm_) -> layer(h_, c_, pm_; rope=rope, cache=cache, layer_kws...),
                h, cond, pos_mask,
            )
        else
            h = layer(h, cond, pos_mask; rope, cache, layer_kws...)
        end
    end

    h = model.norm(h)
    return model.output(h)
end

(model::AdaConditionalTransformer)(tokens::AbstractArray{<:Integer}, conditional; kws...) = model(tokens, (conditional,); kws...)
(model::AdaConditionalTransformer)(tokens::AbstractArray{<:Integer}; kws...) = model(tokens, (); kws...)



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
function _build_pos_mask(tokens::AbstractArray{<:Integer}, conditional_list;
                         conditional_mask_gen=nothing, start_token_id::Int=2)
    if conditional_mask_gen === nothing
        return default_pos_mask(tokens; start_token_id=start_token_id)  # (1,seq,batch)
    else
        cm = ChainRulesCore.ignore_derivatives() do
            conditional_mask_gen(tokens, conditional_list)               # (seq,batch,ncond)
        end
        return conditional_mask_to_pos_mask(cm)                          # (1,seq,batch)
    end
end

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
    n > 1 && model(device(tokens[1:n-1, :]), conditionals; caches, mask=causal_mask, kws...)
    for _ in 1:max_new_tokens
        logits = model(device(tokens[end:end, 1]), conditionals; caches, kws...)
        newtok = sampler(logits[:, end])
        newtok = newtok isa Number ? reshape([newtok], 1, 1) : reshape(newtok, 1, :)
        tokens = vcat(tokens, newtok)
        !isnothing(tokenizer_for_printing) && print(io, decode(tokenizer_for_printing, tokens[end:end] |> cpu, skip_special_tokens = false))
        sum(tokens[end:end]) == end_token && break
    end
    return tokens
end