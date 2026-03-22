import torch
from torch import nn

from einops import rearrange, einsum
from jaxtyping import Bool, Float, Int
from torch import Tensor


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T

    # Xavier/Glorot initialization
    def _init_weight(self):
        std = (2 / (self.in_features + self.out_features)) ** 0.5
        torch.nn.init.trunc_normal_(self.weight, mean=0, std=std, a=-3 * std, b=3 * std)


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))

    def forward(self, token_ids: list[int]) -> torch.Tensor:
        return self.weight[token_ids]

    def _init_weight(self):
        std = 1
        torch.nn.init.trunc_normal_(self.weight, mean=0, std=std, a=-3 * std, b=3 * std)


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype
        self.weight = nn.Parameter(torch.empty(d_model, device=device, dtype=dtype))

    def forward(self, in_features: Float[Tensor, " ... d_model"]) -> torch.Tensor:
        in_dtype = in_features.dtype
        # transform to float32 to prevent overflow
        in_features = in_features.to(torch.float32)
        mse = in_features.pow(2).mean(dim=-1, keepdim=True)
        inv_rms = torch.rsqrt(mse + self.eps)
        res = in_features * inv_rms * self.weight
        return res.to(in_dtype)


def softmax(in_features: Float[Tensor, " ..."], dim: int):
    in_features_max = in_features.max(dim=dim, keepdim=True).values
    x = in_features - in_features_max
    exp_x = torch.exp(x)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)


class Swiglu(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.device = device
        self.dtype = dtype
        self.w1_weight = nn.Parameter(torch.empty(d_ff, d_model, device=device, dtype=dtype))
        self.w2_weight = nn.Parameter(torch.empty(d_model, d_ff, device=device, dtype=dtype))
        self.w3_weight = nn.Parameter(torch.empty(d_ff, d_model, device=device, dtype=dtype))

    # SwiGLU(x) = W2(silu(W1x) * (W3x))
    def forward(self, in_feature: torch.Tensor):
        inner = einsum(in_feature, self.w1_weight, "... d_model, d_ff d_model -> ... d_ff")
        swish = inner * torch.sigmoid(inner)
        data_path = einsum(in_feature, self.w3_weight, "... d_model, d_ff d_model -> ... d_ff")
        swiglu = swish * data_path
        return einsum(swiglu, self.w2_weight, "... d_ff, d_model d_ff -> ... d_model")


class Silu(nn.Module):
    def __init__(
        self,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.device = device
        self.dtype = dtype

    def forward(self, in_feature: torch.Tensor) -> torch.Tensor:
        return in_feature * in_feature.sigmoid()


def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    dk = Q.shape[-1]
    scores = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys") / dk**0.5
    if mask is not None:
        scores = scores.masked_fill(mask == False, float("-inf"))
    attn = softmax(scores, dim=-1)
    return einsum(attn, V, "... queries keys, ... keys d_v -> ... queries d_v")


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        use_mask: bool = True,
        use_rope: bool = False,
        max_seq_len: int = 0,
        theta: float = 10000,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.device = device
        self.dtype = dtype
        self.use_mask = use_mask
        self.dhead = self.dk = self.dv = d_model // num_heads
        self.use_rope = use_rope
        self.rope = None
        if self.use_rope:
            self.rope = Rope(d_k=self.dhead, theta=theta, max_seq_len=max_seq_len, device=device, dtype=dtype)

        self.q = nn.Parameter(torch.empty(d_model, d_model, device=device, dtype=dtype))
        self.k = nn.Parameter(torch.empty(d_model, d_model, device=device, dtype=dtype))
        self.v = nn.Parameter(torch.empty(d_model, d_model, device=device, dtype=dtype))
        self.o = nn.Parameter(torch.empty(d_model, d_model, device=device, dtype=dtype))

    def forward(
        self,
        in_features: Float[Tensor, " ... seq d_in"],
        token_positions: Int[Tensor, " ... sequence_length"] | None = None,
    ) -> Float[Tensor, " ... seq d_out"]:
        # Linear projections
        q = einsum(in_features, self.q, "... seq d_in, d_model d_in -> ... seq d_model")
        k = einsum(in_features, self.k, "... seq d_in, d_model d_in -> ... seq d_model")
        v = einsum(in_features, self.v, "... seq d_in, d_model d_in -> ... seq d_model")

        # Split heads - changed to ... h seq d for easier attention calculation
        q = rearrange(q, "... seq (h d) -> ... h seq d", h=self.num_heads)
        k = rearrange(k, "... seq (h d) -> ... h seq d", h=self.num_heads)
        v = rearrange(v, "... seq (h d) -> ... h seq d", h=self.num_heads)

        if self.use_rope:
            assert token_positions is not None
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        # Scaled dot-product attention
        mask = None
        if self.use_mask:
            seq_len = in_features.shape[-2]
            mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=in_features.device))
            # broadcast to (batch, heads, seq, seq)
            mask = mask.unsqueeze(0).unsqueeze(0)
        attn_out = scaled_dot_product_attention(q, k, v, mask)

        # Merge heads
        attn_out = rearrange(attn_out, "... h seq d -> ... seq (h d)")

        # Final linear projection
        return einsum(attn_out, self.o, "... seq d_v, d_model d_v -> ... seq d_model")


class Rope(nn.Module):
    def __init__(
        self,
        d_k: int,
        theta: float,
        max_seq_len: int,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device
        self.dtype = dtype
        self.theta = theta

        half_d_k = d_k // 2
        freq_seq = torch.arange(half_d_k, device=device, dtype=torch.float32)
        inv_freq = theta ** (-2 * freq_seq / d_k)
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32).unsqueeze(-1)
        angles = positions * inv_freq.unsqueeze(0)

        # Cache all position-dependent trig values once up to max_seq_len.
        self.register_buffer("cos_cache", torch.cos(angles), persistent=False)
        self.register_buffer("sin_cache", torch.sin(angles), persistent=False)

    def forward(
        self,
        x: Float[Tensor, " ... seq_len d_k"],
        token_positions: Float[Tensor, " ... seq_len"],
    ) -> Float[Tensor, " ... seq_len d_k"]:
        token_positions = token_positions.to(torch.long)
        cos = self.cos_cache[token_positions].to(dtype=x.dtype)
        sin = self.sin_cache[token_positions].to(dtype=x.dtype)

        # Group each even/odd feature pair so each pair gets one 2x2 rotation.
        x_pairs = rearrange(x, "... seq_len (d two) -> ... seq_len d two", two=2)
        rope_matrix = torch.stack(
            (
                torch.stack((cos, -sin), dim=-1),
                torch.stack((sin, cos), dim=-1),
            ),
            dim=-2,
        )
        rotated = einsum(rope_matrix, x_pairs, "... seq_len d i j, ... seq_len d j -> ... seq_len d i")
        return rearrange(rotated, "... seq_len d two -> ... seq_len (d two)")


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.ln1 = RMSNorm(d_model=d_model, eps=1e-5, device=device, dtype=dtype)
        self.attn = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            use_mask=True,
            use_rope=True,
            max_seq_len=max_seq_len,
            theta=theta,
            device=device,
            dtype=dtype,
        )
        self.ln2 = RMSNorm(d_model=d_model, eps=1e-5, device=device, dtype=dtype)
        self.ffn = Swiglu(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)

    def forward(
        self,
        x: Float[Tensor, " batch sequence_length d_model"],
        token_positions: Int[Tensor, " batch sequence_length"] | None = None,
    ) -> Float[Tensor, " batch sequence_length d_model"]:
        if token_positions is None:
            seq_len = x.shape[-2]
            token_positions = torch.arange(seq_len, device=x.device)
            token_positions = token_positions.unsqueeze(0).expand(x.shape[0], -1)

        # Pre-norm block: x -> ln1 -> attention -> residual -> ln2 -> SwiGLU -> residual.
        attn_input = self.ln1(x)
        x = x + self.attn(attn_input, token_positions)

        ffn_input = self.ln2(x)
        return x + self.ffn(ffn_input)
