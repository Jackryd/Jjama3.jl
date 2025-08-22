### Layers ###

@concrete struct ConditionalTransformer
    tok_embeddings
    cond_embeddings
    layers
    norm
    output
    rope
end

Flux.@layer ConditionalTransformer

function ConditionalTransformer(cond_embeddings::Tuple,
    vocab_size::Int, dim::Int, n_layers::Int, n_heads::Int, 
    n_kv_heads::Int, max_seq_len::Int, ff_hidden_dim::Int;
    norm_eps=1f-5,
    rope_settings = (theta = 500000f0, use_scaled = false, scale_factor = 8),
    head_dim = dim ÷ n_heads,
    kws...
)
    tok_embeddings = Embedding(vocab_size => dim)
    layers = Tuple(TransformerBlock(dim, n_heads, n_kv_heads, ff_hidden_dim; norm_eps, head_dim, kws...) for _ in 1:n_layers)
    norm = RMSNorm(dim, eps=norm_eps)
    output = Dense(dim => vocab_size, bias=false)
    rope = RoPE(head_dim, max_seq_len * 2; rope_settings...)
    ConditionalTransformer(tok_embeddings, cond_embeddings, layers, norm, output, rope)
end

function ConditionalTransformer(cond_embedding,
    vocab_size::Int, dim::Int, n_layers::Int, n_heads::Int, 
    n_kv_heads::Int, max_seq_len::Int, ff_hidden_dim::Int;
    norm_eps=1f-5,
    rope_settings = (theta = 500000f0, use_scaled = false, scale_factor = 8),
    head_dim = dim ÷ n_heads,
    kws...
)
    ConditionalTransformer((cond_embedding, ), vocab_size, dim, n_layers, n_heads, n_kv_heads, max_seq_len, ff_hidden_dim; norm_eps, rope_settings, head_dim, kws...)
end


function clear_cache!(model::ConditionalTransformer)
    model.pos = 0
    for layer in model.layers
        clear!(layer.attention.cache)
    end
end

### Model ###

function default_conditional_mask(tokens, conditional_list)
    mask = similar(tokens, size(tokens, 1), size(tokens, 2), length(conditional_list))
    mask .= 1
    return mask
end

function (model::ConditionalTransformer)(tokens::AbstractArray{Int}, conditionals::Tuple; 
                                         conditional_list = 1:length(conditionals), conditional_mask_gen = default_conditional_mask, caches = no_kv_cache(model), kws...)
    conditional_mask = Flux.ChainRulesCore.ignore_derivatives() do
        conditional_mask_gen(tokens, conditional_list)
    end # (seq_len, batch, length(conditional_list))
    h = model.tok_embeddings(tokens) # Embedding: (dim, seq_len, batch)
    for (ic, c) in enumerate(conditional_list)
        cond_emb = model.cond_embeddings[c]
        cond, cond_mask = conditionals[ic], conditional_mask[:, :, ic]
        h = h .+ rearrange(cond_emb(cond), (:dim, :batch) --> (:dim, 1, :batch)) .*
                 rearrange(cond_mask, (:seq_len, :batch) --> (1, :seq_len, :batch))
    end
    rope = model.rope[position(caches) .+ (1:size(tokens, 1))]
    for (layer, cache) in zip(model.layers, caches)
        h = layer(h; rope, cache, kws...)
    end
    h = model.norm(h)
    output = model.output(h)
    return output
end

(model::ConditionalTransformer)(tokens::AbstractArray{Int}, conditional; kws...) = model(tokens, (conditional,); kws...)
(model::ConditionalTransformer)(tokens::AbstractArray{Int}; kws...) = model(tokens, (); kws...)

# compat
forward_loss(model::ConditionalTransformer, inputs::AbstractArray, conditionals, targets::AbstractArray; loss_mask = nothing, kws...) = loss(model(inputs, conditionals; kws...), targets, loss_mask = loss_mask) 

### Sampling ###

"""
    generate(model, initial_tokens; max_new_tokens=100, sampler=top_pk_sampler(p=0.5f0, k=5), tokenizer_for_printing=tkn, end_token=128010)

Takes an initial sequence of tokens, and generates new tokens one at a time until the end token is sampled. Uses a KV cache. No batch dim for now.
Runs on CPU by default. If the model is on the GPU (assuming Flux.jl, eg. `model = gpu(model)`), then pass `device = gpu` to `generate` to run on the GPU.

```julia
tkn = llama3_tokenizer()
generate(model, initial_tokens; max_new_tokens=100, sampler=top_pk_sampler(p=0.5f0, k=5), tokenizer_for_printing=tkn, end_token=128010)
```
"""
function generate(
    model::ConditionalTransformer, 
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
    tokens = reshape(initial_tokens, n, b)  # (seq_len, batch=1)
    conditionals = device(conditionals)
    n > 1 && model(device(tokens[1:n-1, :]), conditionals; caches, mask=causal_mask, kws...)
    for i in 1:max_new_tokens
        logits = model(device(tokens[end:end, 1]), conditionals; caches, kws...)
        tokens = [tokens; sampler(logits[:, end])]
        !isnothing(tokenizer_for_printing) && print(io, decode(tokenizer_for_printing, tokens[end:end] |> cpu, skip_special_tokens = false))
        sum(tokens[end:end]) == end_token && break
    end
    return tokens
end