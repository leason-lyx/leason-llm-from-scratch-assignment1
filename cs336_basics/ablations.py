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

from myoperator import (
    MultiheadSelfAttention,
    SwiGLU,
    silu,
    RMSNorm,
    Embedding,
    Linear,
)


class TransformerBlock_noRMSNorm(nn.Module):

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.attn = MultiheadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            theta=theta,
            device=device,
            dtype=dtype,
        )
        self.ffn = SwiGLU(
            d_model=d_model,
            d_ff=d_ff,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        x: Float[Tensor, " ... seq_len d_model"],
    ) -> Float[Tensor, " ... seq_len d_model"]:

        # Apply multi-head self-attention
        attn_output: Float[Tensor, " ... seq_len d_model"] = self.attn.forward(x)

        # Residual connection
        x_res1: Float[Tensor, " ... seq_len d_model"] = x + attn_output

        # Apply feed-forward network
        ffn_output: Float[Tensor, " ... seq_len d_model"] = self.ffn.forward(x_res1)

        # Second residual connection
        output: Float[Tensor, " ... seq_len d_model"] = x_res1 + ffn_output

        return output


class TransformerLM_noRMSNorm(nn.Module):
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
        )

        self.layers = nn.ModuleList(
            [
                TransformerBlock_noRMSNorm(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=context_length,
                    theta=rope_theta,
                    device=device,
                    dtype=dtype,
                )
                for layer_idx in range(num_layers)
            ]
        )

        self.lm_head = Linear(
            in_features=d_model,
            out_features=vocab_size,
            device=device,
            dtype=dtype,
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
        logits: Float[Tensor, "batch_size seq_len vocab_size"] = self.lm_head(x)
        return logits


class TransformerBlock_noPE(nn.Module):

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.attn = MultiheadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            theta=None,
            device=device,
            dtype=dtype,
        )
        self.ln1 = RMSNorm(
            d_model=d_model,
            device=device,
            dtype=dtype,
        )
        self.ffn = SwiGLU(
            d_model=d_model,
            d_ff=d_ff,
            device=device,
            dtype=dtype,
        )
        self.ln2 = RMSNorm(
            d_model=d_model,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        x: Float[Tensor, " ... seq_len d_model"],
    ) -> Float[Tensor, " ... seq_len d_model"]:
        # Apply first RMSNorm
        x_norm1: Float[Tensor, " ... seq_len d_model"] = self.ln1.forward(x)

        # Apply multi-head self-attention without positional embeddings
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


class TransformerLM_noPE(nn.Module):
    @jaxtyped(typechecker=typechecker)
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
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

        self.token_embeddings = Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            device=device,
            dtype=dtype,
        )

        self.layers = nn.ModuleList(
            [
                TransformerBlock_noPE(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=context_length,
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
        )
        self.lm_head = Linear(
            in_features=d_model,
            out_features=vocab_size,
            device=device,
            dtype=dtype,
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


class FFNSiLU(nn.Module):

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.w1 = Linear(
            in_features=d_model,
            out_features=d_ff,
            device=device,
            dtype=dtype,
        )
        self.w2 = Linear(
            in_features=d_ff,
            out_features=d_model,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        x: Float[Tensor, " ... seq_len d_model"],
    ) -> Float[Tensor, " ... seq_len d_model"]:
        return self.w2.forward(silu(self.w1.forward(x)))


class TransformerBlock_siluFFN(nn.Module):

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.attn = MultiheadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            theta=theta,
            device=device,
            dtype=dtype,
        )
        self.ln1 = RMSNorm(
            d_model=d_model,
            device=device,
            dtype=dtype,
        )
        self.ffn = FFNSiLU(
            d_model=d_model,
            d_ff=d_ff,
            device=device,
            dtype=dtype,
        )
        self.ln2 = RMSNorm(
            d_model=d_model,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        x: Float[Tensor, " ... seq_len d_model"],
    ) -> Float[Tensor, " ... seq_len d_model"]:
        # Apply first RMSNorm
        x_norm1: Float[Tensor, " ... seq_len d_model"] = self.ln1.forward(x)

        # Apply multi-head self-attention
        attn_output: Float[Tensor, " ... seq_len d_model"] = self.attn.forward(x_norm1)

        # Residual connection
        x_res1: Float[Tensor, " ... seq_len d_model"] = x + attn_output

        # Apply second RMSNorm
        x_norm2: Float[Tensor, " ... seq_len d_model"] = self.ln2.forward(x_res1)

        # Apply feed-forward network (SiLU, no GLU)
        ffn_output: Float[Tensor, " ... seq_len d_model"] = self.ffn.forward(x_norm2)

        # Second residual connection
        output: Float[Tensor, " ... seq_len d_model"] = x_res1 + ffn_output

        return output


class TransformerLM_siluFFN(nn.Module):
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
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.rope_theta = rope_theta

        d_ff_silu = 4 * d_model
        self.d_ff = d_ff_silu

        self.token_embeddings = Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            device=device,
            dtype=dtype,
        )

        self.layers = nn.ModuleList(
            [
                TransformerBlock_siluFFN(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff_silu,
                    max_seq_len=context_length,
                    theta=rope_theta,
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
        )
        self.lm_head = Linear(
            in_features=d_model,
            out_features=vocab_size,
            device=device,
            dtype=dtype,
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


class TransformerBlock_postnorm(nn.Module):

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.attn = MultiheadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            theta=theta,
            device=device,
            dtype=dtype,
        )
        self.ln1 = RMSNorm(
            d_model=d_model,
            device=device,
            dtype=dtype,
        )
        self.ffn = SwiGLU(
            d_model=d_model,
            d_ff=d_ff,
            device=device,
            dtype=dtype,
        )
        self.ln2 = RMSNorm(
            d_model=d_model,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        x: Float[Tensor, " ... seq_len d_model"],
    ) -> Float[Tensor, " ... seq_len d_model"]:

        # Apply multi-head self-attention
        attn_output: Float[Tensor, " ... seq_len d_model"] = self.attn.forward(x)

        # Residual connection and first RMSNorm
        x_res1: Float[Tensor, " ... seq_len d_model"] = x + attn_output
        x_norm1: Float[Tensor, " ... seq_len d_model"] = self.ln1.forward(x_res1)

        # Apply feed-forward network
        ffn_output: Float[Tensor, " ... seq_len d_model"] = self.ffn.forward(x_norm1)

        # Second residual connection and second RMSNorm
        x_res2: Float[Tensor, " ... seq_len d_model"] = x_res1 + ffn_output
        output: Float[Tensor, " ... seq_len d_model"] = self.ln2.forward(x_res2)

        return output


class TransformerLM_postnorm(nn.Module):
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
        )

        self.layers = nn.ModuleList(
            [
                TransformerBlock_postnorm(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=context_length,
                    theta=rope_theta,
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
        )
        self.lm_head = Linear(
            in_features=d_model,
            out_features=vocab_size,
            device=device,
            dtype=dtype,
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
