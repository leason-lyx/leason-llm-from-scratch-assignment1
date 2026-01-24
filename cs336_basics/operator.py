import torch
from torch import Tensor, LongTensor
import torch.nn as nn
from jaxtyping import jaxtyped, Float, Int, Bool
from einops import rearrange, einsum, reduce, pack, unpack
import math
from beartype import beartype as typechecker


def softmax(x: Float[Tensor, "..."], dim: int = -1) -> Float[Tensor, "..."]:
    """Compute the softmax of the input tensor along dim.

    Args:
        x: Input tensor of arbitrary shape.
        dim: Dimension along which to compute the softmax. Defaults to -1.

    Returns:
        A tensor of the same shape as `x` with softmax applied along the last dimension.
    """
    # Subtract the max for numerical stability
    x_max: Float[Tensor, "... 1"] = torch.max(x, dim=dim, keepdim=True).values
    x_stable: Float[Tensor, "..."] = x - x_max

    exp_x: Float[Tensor, "..."] = torch.exp(x_stable)
    sum_exp_x: Float[Tensor, "... 1"] = torch.sum(exp_x, dim=dim, keepdim=True)
    softmax_x: Float[Tensor, "..."] = exp_x / sum_exp_x
    return softmax_x


def scaled_dot_product_attention(
    Q: Float[Tensor, "b ... seq_len d_k"],
    K: Float[Tensor, "b ... seq_len d_k"],
    V: Float[Tensor, "b ... seq_len d_v"],
    mask: Bool[Tensor, "seq_len seq_len"] | None = None,
) -> Float[Tensor, "b ... d_v"]:
    d_k = Q.shape[-1]
    qk = einsum(Q, K, "b ... q d_k, b ... k d_k -> b ... q k") / math.sqrt(d_k)
    if mask is not None:
        qk_mask: Float[Tensor, "b ... seq_len seq_len"] = qk.masked_fill(
            mask=~mask, value=float("-inf")
        )
    else:
        qk_mask: Float[Tensor, "b ... seq_len seq_len"] = qk
    softmax_value = softmax(qk_mask, dim=-1)
    result: Float[Tensor, "b ... d_v"] = einsum(
        softmax_value, V, "b ... q k, b ... k d_v -> b ... q d_v"
    )
    return result


class Linear(nn.Module):

    @jaxtyped(typechecker=typechecker)
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        weights: Float[Tensor, "d_out d_in"] | None = None,
    ) -> None:
        """
        initialize

        Args:
            in_features: dim of input tensor
            out_features: dim of result tensor
            device: device. Defaults to None.
            dtype: dtype. Defaults to None.
        """
        super().__init__()
        self.in_features: int = in_features
        self.out_features: int = out_features
        if weights is None:
            self.w: Float[Tensor, "d_out d_in"] = nn.Parameter(
                data=torch.empty(
                    size=(out_features, in_features), dtype=dtype, device=device
                )
            )
            std = math.sqrt(2 / (in_features + out_features))
            nn.init.trunc_normal_(tensor=self.w, mean=0, std=std, a=-3 * std, b=3 * std)
        else:
            weights.to(dtype=dtype, device=device)
            self.w: Float[Tensor, "d_out d_in"] = nn.Parameter(weights)
        self.device = self.w.device
        self.dtype = self.w.dtype

    @jaxtyped(typechecker=typechecker)
    def forward(self, x: Float[Tensor, "... d_in"]) -> Float[Tensor, "... d_out"]:
        result: Float[Tensor, "... d_out"] = einsum(
            self.w, x, "d_out d_in,... d_in -> ... d_out"
        )
        return result


class Embedding(nn.Module):

    @jaxtyped(typechecker=typechecker)
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        weights: Float[Tensor, "vocab_size d_model"] | None = None,
    ) -> None:
        """initialize

        Args:
            num_embeddings: Size of the vocabulary
            embedding_dim: Dimension of the embedding vectors
            device: Device to store the parameters on. Defaults to torch.device | None=None.
            dtype: Data type of the parameters. Defaults to torch.dtype | None=None.
        """
        super().__init__()
        self.num_embeddings: int = num_embeddings
        self.embedding_dim: int = embedding_dim
        if weights is None:
            self.embedding: Float[Tensor, "vacab_size d_model"] = nn.Parameter(
                data=torch.empty(
                    size=(num_embeddings, embedding_dim), device=device, dtype=dtype
                )
            )
            nn.init.trunc_normal_(tensor=self.embedding, mean=0, std=1, a=-3, b=-3)
        else:
            weights.to(device=device, dtype=dtype)
            self.embedding: Float[Tensor, "vacab_size d_model"] = nn.Parameter(
                data=weights
            )
        self.device = self.embedding.device
        self.dtype = self.embedding.dtype

    @jaxtyped(typechecker=typechecker)
    def forward(
        self, token_ids: Int[Tensor, "batch_size seq_len"]
    ) -> Float[Tensor, "batch_size seq_len d_model"]:
        result: Float[Tensor, "batch_size seq_len d_model"] = self.embedding[token_ids]
        return result


class RMSNorm(nn.Module):

    @jaxtyped(typechecker=typechecker)
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        weights: Float[Tensor, "d_model"] | None = None,
    ) -> None:
        """Construct the RMSNorm module

        Args:
            d_model: Hidden dimension of the model
            eps: Epsilon value for numerical stability. Defaults to 1e-5.
            device: Device to store the parameters on. Defaults to None.
            dtype: Data type of the parameters. Defaults to None.
        """
        super().__init__()
        self.d_model: int = d_model
        self.eps: float = eps
        if weights is None:
            self.g: Float[Tensor, "d_model"] = nn.Parameter(
                data=torch.ones(size=(d_model,), device=device, dtype=dtype)
            )
        else:
            weights.to(device=device, dtype=dtype)
            self.g: Float[Tensor, "d_model"] = nn.Parameter(data=weights)
        self.device = device
        self.dtype = dtype

    @jaxtyped(typechecker=typechecker)
    def forward(
        self, x: Float[Tensor, "batch_size sequence_length d_model"]
    ) -> Float[Tensor, "batch_size sequence_length d_model"]:
        in_dtype = x.dtype
        x = x.to(dtype=torch.float32)

        # calculate RMS
        mean_sq = reduce(
            x**2,
            "batch_size sequence_length d_model -> batch_size sequence_length 1",
            "mean",
        )
        rms: Float[Tensor, "batch_size sequence_length 1"] = torch.sqrt(
            mean_sq + self.eps
        )
        print(x.size())
        print(rms.size())
        result: Float[Tensor, "batch_size sequence_length d_model"] = x / rms * self.g
        return result.to(in_dtype)


class SwiGLU(nn.Module):
    @jaxtyped(typechecker=typechecker)
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        w1_weight: Float[Tensor, "d_ff d_model"] | None = None,
        w2_weight: Float[Tensor, "d_model d_ff"] | None = None,
        w3_weight: Float[Tensor, "d_ff d_model"] | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        if w1_weight is not None:
            self.w1 = Linear(in_features=d_model, out_features=d_ff, weights=w1_weight)
        else:
            self.w1 = Linear(in_features=d_model, out_features=d_ff)
        if w2_weight is not None:
            self.w2 = Linear(in_features=d_ff, out_features=d_model, weights=w2_weight)
        else:
            self.w2 = Linear(in_features=d_ff, out_features=d_model)
        if w3_weight is not None:
            self.w3 = Linear(in_features=d_model, out_features=d_ff, weights=w3_weight)
        else:
            self.w3 = Linear(in_features=d_model, out_features=d_ff)

    @jaxtyped(typechecker=typechecker)
    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        x1: Float[Tensor, "... d_ff"] = self.w1.forward(x)
        x3: Float[Tensor, "... d_ff"] = self.w3.forward(x)
        silu: Float[Tensor, "... d_ff"] = x1 * torch.sigmoid(x1)
        dot_product: Float[Tensor, "... d_ff"] = torch.mul(
            x3, silu
        )  # element-wise product
        result: Float[Tensor, "... d_model"] = self.w2.forward(dot_product)
        return result


class RotaryPositionalEmbedding(nn.Module):

    @jaxtyped(typechecker=typechecker)
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device
        inv_freq: Float[Tensor, "d_k2"] = theta ** (
            -torch.arange(start=0, end=d_k, step=2, device=device, dtype=torch.float)
            / d_k
        )
        pos: Float[Tensor, "seq"] = torch.arange(
            max_seq_len, device=device, dtype=torch.float
        )
        angles: Float[Tensor, "seq d_k2"] = rearrange(pos, "seq -> seq 1") * rearrange(
            inv_freq, "d_k2 -> 1 d_k2"
        )
        cos: Float[Tensor, "seq d_k2"] = angles.cos()
        sin: Float[Tensor, "seq d_k2"] = angles.sin()
        self.register_buffer(name="cos", tensor=cos, persistent=False)
        self.register_buffer(name="sin", tensor=sin, persistent=False)

    @jaxtyped(typechecker=typechecker)
    def forward(
        self,
        x: Float[Tensor, " ... seq_len d_k"],
        token_positions: Int[Tensor, " ... seq"],
    ) -> Float[Tensor, " ... seq_len d_k"]:
        cos: Float[Tensor, "seq_len d_k2"] = self.cos[token_positions]  # type: ignore
        sin: Float[Tensor, "seq_len d_k2"] = self.sin[token_positions]  # type: ignore

        x_pair: Float[Tensor, "... seq_len d_k2 two"] = rearrange(
            x, "... seq_len (d_k2 two) -> ... seq_len d_k2 two", two=2
        )
        x_even: Float[Tensor, "... seq_len d_k2"] = x_pair[..., 0]
        x_odd: Float[Tensor, "... seq_len d_k2"] = x_pair[..., 1]

        out_even: Float[Tensor, "... seq_len d_k2"] = x_even * cos - x_odd * sin
        out_odd: Float[Tensor, "... seq_len d_k2"] = x_even * sin + x_odd * cos

        out_pair = torch.stack((out_even, out_odd), dim=-1)
        out = rearrange(out_pair, "... seq_len d_k2 two -> ... seq_len (d_k2 two)")

        return out
