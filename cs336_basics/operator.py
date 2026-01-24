import torch
from torch import Tensor
import torch.nn as nn
from jaxtyping import Float
from einops import rearrange, einsum
import math


class Linear(torch.nn.Module):

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
            out_features: dim of output tensor
            device: device. Defaults to None.
            dtype: dtype. Defaults to None.
        """
        super().__init__()
        if weights is None:
            self.w: Float[Tensor, "d_out d_in"] = nn.Parameter(
                torch.empty(
                    size=(out_features, in_features), dtype=dtype, device=device
                )
            )
            std = math.sqrt(2 / (in_features + out_features))
            nn.init.trunc_normal_(tensor=self.w, mean=0, std=std, a=-3 * std, b=3 * std)
        else:
            weights.to(dtype=dtype, device=device)
            self.w: Float[Tensor, "d_out d_in"] = nn.Parameter(weights)

    def forward(self, x: Float[Tensor, "... d_in"]) -> Float[Tensor, "... d_out"]:
        """
        Apply the linear transformation to the input

        Args:
            x: input tensor

        Returns:
            output tensor
        """
        output: Float[Tensor, "... d_out"] = einsum(
            self.w, x, "d_out d_in,... d_in -> ... d_out"
        )
        return output
