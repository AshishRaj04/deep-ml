def gen_text(prompt: str, n_tokens_to_generate: int = 40):
    # Your code here
    np.random.seed(42)
    encoder, hparams, params = load_encoder_hparams_and_params()

    token = encoder.encode(prompt)
    generated = []

    n_head = hparams["n_head"]
    d_model = params["wte"].shape[-1]
    d_head = d_model // n_head

    for _ in range(n_tokens_to_generate):
        L = len(token)

        te = params["wte"][token]
        pe = params["wpe"][:L]
        x = te + pe

        for block in params["blocks"]:

            # LayerNorm 1
            scale, shift = block["ln_1"].values()
            mean = np.mean(x, axis=-1, keepdims=True)
            var = np.var(x, axis=-1, keepdims=True)
            z = scale * ((x - mean) / np.sqrt(var + 1e-5)) + shift

            # Multi-Head Attention
            c_attn = block["attn"]["c_attn"]
            qkv = z @ c_attn["w"] + c_attn["b"]

            q, k, v = np.split(qkv, 3, axis=-1)

            q = q.reshape(L, n_head, d_head).transpose(1, 0, 2)
            k = k.reshape(L, n_head, d_head).transpose(1, 0, 2)
            v = v.reshape(L, n_head, d_head).transpose(1, 0, 2)

            scores = q @ k.transpose(0, 2, 1) / np.sqrt(d_head)

            # Causal mask
            mask = np.triu(np.ones((L, L)) * (-1e10), k=1)
            scores = scores + mask

            # Softmax
            scores = scores - np.max(scores, axis=-1, keepdims=True)
            attn = np.exp(scores)
            attn = attn / np.sum(attn, axis=-1, keepdims=True)

            # Attention weighted values
            h = attn @ v

            # Concatenate heads
            h = h.transpose(1, 0, 2).reshape(L, d_model)

            # Attention output projection
            c_proj = block["attn"]["c_proj"]
            attn_out = h @ c_proj["w"] + c_proj["b"]

            # Residual connection
            x = x + attn_out

            # LayerNorm 2
            scale, shift = block["ln_2"].values()
            mean = np.mean(x, axis=-1, keepdims=True)
            var = np.var(x, axis=-1, keepdims=True)
            z = scale * ((x - mean) / np.sqrt(var + 1e-5)) + shift

            # Feed Forward Network
            c_fc = block["mlp"]["c_fc"]
            h = z @ c_fc["w"] + c_fc["b"]

            # GELU
            h = 0.5 * h * (
                1 + np.tanh(
                    np.sqrt(2 / np.pi) * (h + 0.044715 * h**3)
                )
            )

            c_proj = block["mlp"]["c_proj"]
            mlp_out = h @ c_proj["w"] + c_proj["b"]

            # Residual connection
            x = x + mlp_out

        # Final LayerNorm
        scale, shift = params["ln_f"].values()
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        x = scale * ((x - mean) / np.sqrt(var + 1e-5)) + shift

        # Logits from tied embedding weights
        logits = x @ params["wte"].T

        # Greedy decoding: last token position
        next_token = int(np.argmax(logits[-1]))

        token = np.append(token, next_token)
        generated.append(next_token)

    return encoder.decode(generated)

def load_encoder_hparams_and_params(model_size: str = "124M", models_dir: str = "models"):
    class DummyBPE:
        def __init__(self):
            self.encoder_dict = {"hello": 1, "world": 2, "<UNK>": 0}

        def encode(self, text: str):
            tokens = text.strip().split()
            return [self.encoder_dict.get(token, self.encoder_dict["<UNK>"]) for token in tokens]

        def decode(self, token_ids: list):
            reversed_dict = {v: k for k, v in self.encoder_dict.items()}
            return " ".join([reversed_dict.get(tok_id, "<UNK>") for tok_id in token_ids])

    hparams = {
        "n_ctx": 1024,
        "n_head": 2
    }

    params = {
        "wte": np.random.rand(3, 10),
        "wpe": np.random.rand(1024, 10),
        "blocks": [{
            "mlp": {
                "c_fc": {"w": np.random.rand(10, 20), "b": np.random.rand(20)},
                "c_proj": {"w": np.random.rand(20, 10), "b": np.random.rand(10)}
            },
            "attn": {
                "c_attn": {"w": np.random.rand(10, 30), "b": np.random.rand(30)},
                "c_proj": {"w": np.random.rand(10, 10), "b": np.random.rand(10)}
            },
            "ln_1": {"g": np.ones(10), "b": np.zeros(10)},
            "ln_2": {"g": np.ones(10), "b": np.zeros(10)},
        }],
        "ln_f": {
            "g": np.ones(10),
            "b": np.zeros(10),
        }
    }

    encoder = DummyBPE()
    return encoder, hparams, params