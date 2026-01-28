import os
import typing
import torch
from torch import Tensor, LongTensor
import torch.nn as nn
import numpy.typing as npt
import numpy as np
from collections.abc import Callable, Iterable
from jaxtyping import jaxtyped, Float, Int, Bool
from einops import rearrange, einsum, reduce
import math
from beartype import beartype as typechecker


@jaxtyped(typechecker=typechecker)
def silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    """SiLU activation function.

    Args:
        in_features: Input tensor of arbitrary shape.

    Returns:
        A tensor of the same shape as `in_features` with SiLU applied element-wise.
    """
    return in_features * torch.sigmoid(in_features)


@jaxtyped(typechecker=typechecker)
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


@jaxtyped(typechecker=typechecker)
def scaled_dot_product_attention(
    Q: Float[Tensor, "... q_len d_k"],
    K: Float[Tensor, "... k_len d_k"],
    V: Float[Tensor, "... k_len d_v"],
    mask: Bool[Tensor, "... q_len k_len"] | None = None,
) -> Float[Tensor, "... q_len d_v"]:
    d_k = Q.shape[-1]
    qk = einsum(Q, K, "... q d_k,... k d_k -> ... q k") / math.sqrt(d_k)
    if mask is not None:
        qk_mask: Float[Tensor, "... q_len k_len"] = qk.masked_fill(
            mask=~mask, value=float("-inf")
        )
    else:
        qk_mask: Float[Tensor, "... q_len k_len"] = qk
    softmax_value = softmax(qk_mask, dim=-1)
    result: Float[Tensor, "... q_len d_v"] = einsum(
        softmax_value, V, "... q k, ... k d_v -> ... q d_v"
    )
    return result


@jaxtyped(typechecker=typechecker)
def cross_entropy(
    logits: Float[Tensor, "... seq_len vocab_size"], targets: Int[Tensor, "... seq_len"]
) -> Float[Tensor, "..."]:
    """
    Returns:
        mean cross-entropy loss across seq_len for each example in the batch
    """
    logits_max: Float[Tensor, "... seq_len 1"] = torch.max(
        logits, dim=-1, keepdim=True
    ).values
    logits_stable: Float[Tensor, "... seq_len vocab_size"] = logits - logits_max
    logits_expsum: Float[Tensor, "... seq_len 1"] = torch.exp(logits_stable).sum(
        dim=-1, keepdim=True
    )
    log_probs: Float[Tensor, "... seq_len vocab_size"] = logits_stable - torch.log(
        logits_expsum
    )
    result: Float[Tensor, "..."] = rearrange(
        -torch.gather(
            log_probs, dim=-1, index=rearrange(targets, "... seq_len -> ... seq_len 1")
        ),
        "... seq_len 1 -> ... seq_len",
    ).mean(dim=-1)
    return result


def lr_cosine_schedule(
    t: int, lr_max: float, lr_min: float, T_warmup: int, T_c: int
) -> float:
    if t < T_warmup:
        return lr_max * t / T_warmup
    elif t <= T_c:
        return lr_min + 0.5 * (lr_max - lr_min) * (
            1 + math.cos(math.pi * (t - T_warmup) / (T_c - T_warmup))
        )
    else:
        return lr_min


def get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str | torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset: 1D numpy array of integer token IDs in the dataset.
        batch_size: Desired batch size to sample.
        context_length: Desired context length of each sampled example.
        device: PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    n = len(dataset)
    indices = np.random.randint(0, n - context_length, size=batch_size)
    input_sequences = np.stack([dataset[i : i + context_length] for i in indices])
    labels = np.stack([dataset[i + 1 : i + context_length + 1] for i in indices])
    input_tensor = torch.tensor(input_sequences, dtype=torch.long, device=device)
    labels_tensor = torch.tensor(labels, dtype=torch.long, device=device)
    return input_tensor, labels_tensor


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float):
    eps = 1e-6
    params = [param for param in parameters if param.grad is not None]
    if not params:
        return

    total_norm_sq = torch.zeros((), device=params[0].grad.device)  # type: ignore
    for param in params:
        grad = param.grad.detach()  # type: ignore
        if grad.is_sparse:
            grad = grad.coalesce()
            grad_norm = grad._values().norm(2)
        else:
            grad_norm = grad.norm(2)
        total_norm_sq = total_norm_sq + grad_norm.pow(2)
    total_norm = total_norm_sq.sqrt()

    if total_norm > max_l2_norm:
        clip_coef = max_l2_norm / (total_norm + eps)
        for param in params:
            param.grad.detach().mul_(clip_coef)  # type: ignore


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
) -> None:
    """
    should dump all the state from the first three parameters into the file-like object out.
    """
    model_weights: dict = model.state_dict()
    optimizer_state: dict = optimizer.state_dict()
    checkpoint = {
        "model_state_dict": model_weights,
        "optimizer_state_dict": optimizer_state,
        "iteration": iteration,
    }
    torch.save(checkpoint, out)


def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    load a checkpoint from the file-like object src
    and restore the state of the model and optimizer.
    """
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["iteration"]


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
            weights = weights.to(dtype=dtype, device=device)
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
            nn.init.trunc_normal_(tensor=self.embedding, mean=0, std=1, a=-3, b=3)
        else:
            weights = weights.to(device=device, dtype=dtype)
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
            weights = weights.to(device=device, dtype=dtype)
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
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        if w1_weight is not None:
            self.w1 = Linear(
                in_features=d_model,
                out_features=d_ff,
                weights=w1_weight,
                device=device,
                dtype=dtype,
            )
        else:
            self.w1 = Linear(
                in_features=d_model, out_features=d_ff, device=device, dtype=dtype
            )
        if w2_weight is not None:
            self.w2 = Linear(
                in_features=d_ff,
                out_features=d_model,
                weights=w2_weight,
                device=device,
                dtype=dtype,
            )
        else:
            self.w2 = Linear(
                in_features=d_ff, out_features=d_model, device=device, dtype=dtype
            )
        if w3_weight is not None:
            self.w3 = Linear(
                in_features=d_model,
                out_features=d_ff,
                weights=w3_weight,
                device=device,
                dtype=dtype,
            )
        else:
            self.w3 = Linear(
                in_features=d_model, out_features=d_ff, device=device, dtype=dtype
            )

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
        if d_k % 2 != 0:
            raise ValueError("d_k must be even for rotary positional embedding.")
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
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")
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
        weights: dict[str, Tensor] | None = None,
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
            weights=weights["ln1.weight"] if weights is not None else None,
        )
        self.attn = MultiheadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            theta=theta,
            device=device,
            dtype=dtype,
            q_proj_weight=(
                weights["attn.q_proj.weight"] if weights is not None else None
            ),
            k_proj_weight=(
                weights["attn.k_proj.weight"] if weights is not None else None
            ),
            v_proj_weight=(
                weights["attn.v_proj.weight"] if weights is not None else None
            ),
            o_proj_weight=(
                weights["attn.output_proj.weight"] if weights is not None else None
            ),
        )
        self.ln2 = RMSNorm(
            d_model=d_model,
            device=device,
            dtype=dtype,
            weights=(weights["ln2.weight"] if weights is not None else None),
        )
        self.ffn = SwiGLU(
            d_model=d_model,
            d_ff=d_ff,
            w1_weight=(weights["ffn.w1.weight"] if weights is not None else None),
            w2_weight=(weights["ffn.w2.weight"] if weights is not None else None),
            w3_weight=(weights["ffn.w3.weight"] if weights is not None else None),
            device=device,
            dtype=dtype,
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
        weights: dict[str, Tensor] | None = None,
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
            weights=weights["token_embeddings.weight"] if weights is not None else None,
        )

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=context_length,
                    theta=rope_theta,
                    weights=(
                        {
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
                            "ffn.w1.weight": weights[
                                f"layers.{layer_idx}.ffn.w1.weight"
                            ],
                            "ffn.w2.weight": weights[
                                f"layers.{layer_idx}.ffn.w2.weight"
                            ],
                            "ffn.w3.weight": weights[
                                f"layers.{layer_idx}.ffn.w3.weight"
                            ],
                            "ln2.weight": weights[f"layers.{layer_idx}.ln2.weight"],
                        }
                        if weights is not None
                        else None
                    ),
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
            weights=weights["ln_final.weight"] if weights is not None else None,
        )
        self.lm_head = Linear(
            in_features=d_model,
            out_features=vocab_size,
            device=device,
            dtype=dtype,
            weights=weights["lm_head.weight"] if weights is not None else None,
        )

    @jaxtyped(typechecker=typechecker)
    def forward(
        self,
        in_indices: Int[Tensor, "batch_size seq_len"],
    ) -> Float[Tensor, "batch_size seq_len vocab_size"]:
        seq_len = in_indices.shape[-1]
        if seq_len > self.context_length:
            raise ValueError("sequence_length exceeds context_length.")

        x: Float[Tensor, "batch_size seq_len d_model"] = self.token_embeddings(
            token_ids=in_indices
        )
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        logits: Float[Tensor, "batch_size seq_len vocab_size"] = self.lm_head(x)
        return logits


class SGD(torch.optim.Optimizer):

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
    ) -> None:
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Callable | None = None):  # type: ignore
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # get learning rate
            for param in group["params"]:
                if param.grad is None:
                    continue
                state = self.state[param]
                t = state.get("t", 0)
                grad = param.grad.data
                param.data -= lr / math.sqrt(t + 1) * grad
                state["t"] = t + 1

        return loss


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    def step(self, closure: Callable | None = None):  # type: ignore
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad.data
                state = self.state[param]

                if "step" not in state:
                    state["step"] = 0
                    state["first_moment"] = torch.zeros_like(param.data)
                    state["second_moment"] = torch.zeros_like(param.data)

                state["step"] += 1
                first_moment = state["first_moment"]
                second_moment = state["second_moment"]
                first_moment.mul_(beta1).add_(grad, alpha=1 - beta1)
                second_moment.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                adjusted_lr = lr * math.sqrt(bias_correction2) / bias_correction1
                denom = torch.sqrt(second_moment).add_(eps)
                param.data.addcdiv_(first_moment, denom, value=-adjusted_lr)

                if weight_decay != 0:
                    param.data.add_(param.data, alpha=-lr * weight_decay)

        return loss


class LRCosineScheduler(torch.optim.lr_scheduler.LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        lr_max: float,
        lr_min: float,
        T_warmup: int,
        T_c: int,
        last_epoch: int = -1,
    ) -> None:
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.T_warmup = T_warmup
        self.T_c = T_c
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        t = self.last_epoch
        lr = lr_cosine_schedule(t, self.lr_max, self.lr_min, self.T_warmup, self.T_c)
        return [lr for _ in self.optimizer.param_groups]


def test_sgd(lr: float = 1e-1) -> None:
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr=lr)
    for t in range(10):
        opt.zero_grad()  # Reset the gradients for all learnable parameters.
        loss = (weights**2).mean()  # Compute a scalar loss value.
        print(loss.cpu().item())
        loss.backward()  # Run backward pass, which computes gradients.
        opt.step()  # Run optimizer step.


if __name__ == "__main__":
    print("Testing SGD optimizer, learning rate=1e1")
    test_sgd(lr=1e1)
    print("Testing SGD optimizer, learning rate=1e2")
    test_sgd(lr=1e2)
    print("Testing SGD optimizer, learning rate=1e3")
    test_sgd(lr=1e3)
