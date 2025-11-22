
# === AdaConditionalTransformer ===
function default_pos_mask(tokens::AbstractArray{<:Integer}; start_token_id::Integer)
    is_start = tokens .== start_token_id
    cumsum_start = cumsum(is_start, dims=1)
    m = Float32.(cumsum_start .> 0)
    return reshape(m, 1, size(tokens, 1), size(tokens, 2))
end

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

function (model::AdaConditionalTransformer)(
    tokens::AbstractArray{Int},
    conditionals::Tuple;
    conditional_list = 1:length(conditionals),
    conditional_mask_gen = default_conditional_mask,  # currently unused for Ada
    caches = no_kv_cache(model),
    pos_mask = nothing,
    start_token_id = 2,  # assumes '>' is token 2 in your vocab "<>"
    kws...
)
    # Token embeddings: (dim, seq_len, batch)
    h = model.tok_embeddings(tokens)

    # Sum all conditioning embeddings into a single (dim, batch) vector
    cond = model.cond_embeddings[first(conditional_list)](conditionals[first(conditional_list)])
    for (ic, idx) in enumerate(conditional_list)
        ic == 1 && continue
        cond .+= model.cond_embeddings[idx](conditionals[ic])
    end

    # Position-dependent conditioning mask:
    # if caller didn't provide one, derive it from the tokens (train-time behaviour)
    if isnothing(pos_mask)
        pos_mask = default_pos_mask(tokens; start_token_id=start_token_id)
    end

    # RoPE slice based on current cache position
    rope = model.rope[position(caches) .+ (1:size(tokens, 1))]

    # Pass cond + pos_mask into each AdaTransformerBlock
    for (layer, cache) in zip(model.layers, caches)
        h = layer(h, cond, pos_mask; rope, cache, kws...)
    end

    h = model.norm(h)
    output = model.output(h)
    return output
end


forward_loss(model::AdaConditionalTransformer, inputs::AbstractArray, conditionals, targets::AbstractArray; loss_mask = nothing, kws...) = 
    loss(model(inputs, conditionals; kws...), targets, loss_mask = loss_mask)

function generate(
    model::AdaConditionalTransformer,
    initial_tokens::AbstractArray{<:Integer},  # (seq_len, batch)
    conditionals;
    io = stdout,
    max_new_tokens = 100,
    sampler::Function = argmax_sampler,
    tokenizer_for_printing = nothing,
    end_token = 128010,
    caches = kv_cache(model, 1024, 1),
    device = identity,
    start_token_id = 2,  # '>' token
    kws...
)
    n, b = size(initial_tokens, 1), size(initial_tokens, 2)
    @assert b == 1 "generate currently assumes batch size 1"
    tokens = reshape(initial_tokens, n, b)
    conditionals = device(conditionals)

    # 1) Prefill with the whole prefix (if any)
    if n > 1
        # Auto pos_mask from tokens; this matches training behaviour
        model(device(tokens[1:n-1, :]), conditionals;
              caches, mask=causal_mask, start_token_id=start_token_id, kws...)
    end

    # 2) Generate new tokens, forcing full conditioning (pos_mask = 1)
    pos_mask_single = device(ones(Float32, 1, 1, 1))

    for i in 1:max_new_tokens
        last_tok = device(reshape(tokens[end, :], 1, b))
        logits = model(last_tok, conditionals;
                       caches, pos_mask = pos_mask_single,  # after '>', always conditioned
                       mask = causal_mask, start_token_id = start_token_id, kws...)

        new_token = sampler(logits[:, end])
        tokens = vcat(tokens, reshape([new_token], 1, b))

        if !isnothing(tokenizer_for_printing)
            print(io, decode(tokenizer_for_printing, [new_token] |> cpu, skip_special_tokens = false))
        end
        new_token == end_token && break
    end

    return tokens
end