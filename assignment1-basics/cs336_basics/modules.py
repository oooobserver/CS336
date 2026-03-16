import torch
from torch import nn

from einops import rearrange, einsum
from jaxtyping import Bool, Float
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
        mse = in_features.pow(2).mean(dim=-1, keepdim=True)
        inv_rms = torch.rsqrt(mse + self.eps)
        return in_features * inv_rms * self.weight


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
        self.q = nn.Parameter(torch.empty(d_model, d_model, device=device, dtype=dtype))
        self.k = nn.Parameter(torch.empty(d_model, d_model, device=device, dtype=dtype))
        self.v = nn.Parameter(torch.empty(d_model, d_model, device=device, dtype=dtype))
        self.o = nn.Parameter(torch.empty(d_model, d_model, device=device, dtype=dtype))

    def forward(self, in_features: Float[Tensor, " ... seq d_in"]) -> Float[Tensor, " ... seq d_out"]:
        # Linear projections
        q = einsum(in_features, self.q, "... seq d_in, d_model d_in -> ... seq d_model")
        k = einsum(in_features, self.k, "... seq d_in, d_model d_in -> ... seq d_model")
        v = einsum(in_features, self.v, "... seq d_in, d_model d_in -> ... seq d_model")

        # Split heads - changed to ... h seq d for easier attention calculation
        q = rearrange(q, "... seq (h d) -> ... h seq d", h=self.num_heads)
        k = rearrange(k, "... seq (h d) -> ... h seq d", h=self.num_heads)
        v = rearrange(v, "... seq (h d) -> ... h seq d", h=self.num_heads)

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
        max_seq_len: int,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device
        self.dtype = dtype
        self.theta = nn.Parameter(torch.empty(1, device=device, dtype=dtype))
