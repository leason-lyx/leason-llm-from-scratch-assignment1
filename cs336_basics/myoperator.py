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


class MultiheadSelfAttention(nn.Module):
    @jaxtyped(typechecker=typechecker)
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        theta: float | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        q_proj_weight: Float[Tensor, "hd_k d_model"] | None = None,
        k_proj_weight: Float[Tensor, "hd_k d_model"] | None = None,
        v_proj_weight: Float[Tensor, "hd_v d_model"] | None = None,
        o_proj_weight: Float[Tensor, "d_model hd_v"] | None = None,
    ) -> None:
        """
        Args:
            d_model: dimension of the model
            num_heads: number of attention heads
            max_seq_len: maximum sequence length
            theta: RoPE theta. Defaults to None. if None, no RoPE is applied.
            device: device. Defaults to None.
            dtype: data type. Defaults to None.
            q_proj_weight: query projection weights. Defaults to None.
            k_proj_weight: key projection weights. Defaults to None.
            v_proj_weight: value projection weights. Defaults to None.
            o_proj_weight: output projection weights. Defaults to None.
        """
        super().__init__()
        d_k = d_model // num_heads
        d_v = d_k

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_k
        self.d_v = d_v

        if q_proj_weight is None:
            self.q_proj = Linear(
                in_features=d_model, out_features=d_model, device=device, dtype=dtype
            )
        else:
            self.q_proj = Linear(
                in_features=d_model,
                out_features=d_model,
                device=device,
                dtype=dtype,
                weights=q_proj_weight,
            )
        if k_proj_weight is None:
            self.k_proj = Linear(
                in_features=d_model, out_features=d_model, device=device, dtype=dtype
            )
        else:
            self.k_proj = Linear(
                in_features=d_model,
                out_features=d_model,
                device=device,
                dtype=dtype,
                weights=k_proj_weight,
            )
        if v_proj_weight is None:
            self.v_proj = Linear(
                in_features=d_model, out_features=d_model, device=device, dtype=dtype
            )
        else:
            self.v_proj = Linear(
                in_features=d_model,
                out_features=d_model,
                device=device,
                dtype=dtype,
                weights=v_proj_weight,
            )
        if o_proj_weight is None:
            self.output_proj = Linear(
                in_features=d_model, out_features=d_model, device=device, dtype=dtype
            )
        else:
            self.output_proj = Linear(
                in_features=d_model,
                out_features=d_model,
                device=device,
                dtype=dtype,
                weights=o_proj_weight,
            )

        if theta is not None:
            self.rope = RotaryPositionalEmbedding(
                theta=theta, d_k=d_k, max_seq_len=max_seq_len, device=device
            )
        else:
            self.rope = None

    @jaxtyped(typechecker=typechecker)
    def forward(
        self,
        x: Float[Tensor, " ... seq_len d_model"],
        token_positions: Int[Tensor, " ... seq_len"] | None = None,
    ) -> Float[Tensor, " ... seq_len d_model"]:
        seq_len = x.shape[-2]
        if token_positions is None:
            token_positions = torch.arange(seq_len, device=x.device)
        token_positions = token_positions.to(device=x.device, dtype=torch.long)

        q: Float[Tensor, " ... seq_len d_model"] = self.q_proj.forward(x)
        k: Float[Tensor, " ... seq_len d_model"] = self.k_proj.forward(x)
        v: Float[Tensor, " ... seq_len d_model"] = self.v_proj.forward(x)

        q = rearrange(
            q,
            "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k",
            num_heads=self.num_heads,
        )
        k = rearrange(
            k,
            "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k",
            num_heads=self.num_heads,
        )
        v = rearrange(
            v,
            "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k",
            num_heads=self.num_heads,
        )

        if self.rope is not None:
            q = self.rope.forward(x=q, token_positions=token_positions)
            k = self.rope.forward(x=k, token_positions=token_positions)

        causal_mask: Bool[Tensor, "seq_len seq_len"] = torch.tril(
            torch.ones((seq_len, seq_len), device=x.device, dtype=torch.bool)
        )
        attn: Float[Tensor, " ... num_heads seq_len d_k"] = (
            scaled_dot_product_attention(Q=q, K=k, V=v, mask=causal_mask)
        )

        out: Float[Tensor, " ... seq_len d_model"] = rearrange(
            attn,
            "... num_heads seq_len d_k -> ... seq_len (num_heads d_k)",
            num_heads=self.num_heads,
        )
        result: Float[Tensor, " ... seq_len d_model"] = self.output_proj.forward(out)
        return result


class TransformerBlock(nn.Module):

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        weights: dict[str, Tensor],
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """_summary_

        Args:
            d_model: _description_
            num_heads: _description_
            d_ff: _description_
            max_seq_len: _description_
            theta: _description_
            weights: Args:
            d_model (int): The dimensionality of the Transformer block input.
            num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
                evenly divisible by `num_heads`.
            d_ff (int): Dimensionality of the feed-forward inner layer.
            max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
            theta (float): RoPE parameter.
            weights (dict[str, Tensor]):
            State dict of our reference implementation.
            The keys of this dictionary are:
            - `attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is (d_model, d_model).
            - `ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
        """
        super().__init__()
        self.ln1 = RMSNorm(
            d_model=d_model,
            device=device,
            dtype=dtype,
            weights=weights["ln1.weight"],
        )
        self.attn = MultiheadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            theta=theta,
            device=device,
            dtype=dtype,
            q_proj_weight=weights["attn.q_proj.weight"],
            k_proj_weight=weights["attn.k_proj.weight"],
            v_proj_weight=weights["attn.v_proj.weight"],
            o_proj_weight=weights["attn.output_proj.weight"],
        )
        self.ln2 = RMSNorm(
            d_model=d_model,
            device=device,
            dtype=dtype,
            weights=weights["ln2.weight"],
        )
        self.ffn = SwiGLU(
            d_model=d_model,
            d_ff=d_ff,
            w1_weight=weights["ffn.w1.weight"],
            w2_weight=weights["ffn.w2.weight"],
            w3_weight=weights["ffn.w3.weight"],
        )

    def forward(
        self,
        x: Float[Tensor, " ... seq_len d_model"],
    ) -> Float[Tensor, " ... seq_len d_model"]:
        """Forward pass of the Transformer block.

        Args:
            x: Input tensor of shape (..., seq_len, d_model)
            token_positions: Tensor of shape (..., seq_len) indicating the position of each token.

        Returns:
            Output tensor of shape (..., seq_len, d_model)
        """
        # Apply first RMSNorm
        x_norm1: Float[Tensor, " ... seq_len d_model"] = self.ln1.forward(x)

        # Apply multi-head self-attention
        attn_output: Float[Tensor, " ... seq_len d_model"] = self.attn.forward(x_norm1)

        # Residual connection
        x_res1: Float[Tensor, " ... seq_len d_model"] = x + attn_output

        # Apply second RMSNorm
        x_norm2: Float[Tensor, " ... seq_len d_model"] = self.ln2.forward(x_res1)

        # Apply feed-forward network
        ffn_output: Float[Tensor, " ... seq_len d_model"] = self.ffn.forward(x_norm2)

        # Second residual connection
        output: Float[Tensor, " ... seq_len d_model"] = x_res1 + ffn_output

        return output


class TransformerLM(nn.Module):
    @jaxtyped(typechecker=typechecker)
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        weights: dict[str, Tensor],
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta

        self.token_embeddings = Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            device=device,
            dtype=dtype,
            weights=weights["token_embeddings.weight"],
        )

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=context_length,
                    theta=rope_theta,
                    weights={
                        "attn.q_proj.weight": weights[
                            f"layers.{layer_idx}.attn.q_proj.weight"
                        ],
                        "attn.k_proj.weight": weights[
                            f"layers.{layer_idx}.attn.k_proj.weight"
                        ],
                        "attn.v_proj.weight": weights[
                            f"layers.{layer_idx}.attn.v_proj.weight"
                        ],
                        "attn.output_proj.weight": weights[
                            f"layers.{layer_idx}.attn.output_proj.weight"
                        ],
                        "ln1.weight": weights[f"layers.{layer_idx}.ln1.weight"],
                        "ffn.w1.weight": weights[f"layers.{layer_idx}.ffn.w1.weight"],
                        "ffn.w2.weight": weights[f"layers.{layer_idx}.ffn.w2.weight"],
                        "ffn.w3.weight": weights[f"layers.{layer_idx}.ffn.w3.weight"],
                        "ln2.weight": weights[f"layers.{layer_idx}.ln2.weight"],
                    },
                    device=device,
                    dtype=dtype,
                )
                for layer_idx in range(num_layers)
            ]
        )

        self.ln_final = RMSNorm(
            d_model=d_model,
            device=device,
            dtype=dtype,
            weights=weights["ln_final.weight"],
        )
        self.lm_head = Linear(
            in_features=d_model,
            out_features=vocab_size,
            device=device,
            dtype=dtype,
            weights=weights["lm_head.weight"],
        )

    @jaxtyped(typechecker=typechecker)
    def forward(
        self,
        in_indices: Int[Tensor, "batch_size seq_len"],
    ) -> Float[Tensor, "batch_size seq_len vocab_size"]:
        seq_len = in_indices.shape[-1]
        if seq_len > self.context_length:
            raise ValueError("sequence_length exceeds context_length.")

        x: Float[Tensor, "batch_size seq_len d_model"] = self.token_embeddings.forward(
            token_ids=in_indices
        )
        for layer in self.layers:
            x = layer.forward(x)
        x = self.ln_final.forward(x)
        logits: Float[Tensor, "batch_size seq_len vocab_size"] = self.lm_head.forward(x)
        return logits
