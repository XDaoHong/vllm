"""Custom normalization layers."""
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch_npu

from vllm import _custom_ops as ops


class RMSNorm(nn.Module):
    """Root mean square normalization.

    Computes x -> w * x / sqrt(E[x^2] + eps) where w is the learned weight.
    Refer to https://arxiv.org/abs/1910.07467
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x, residual = None):
        if residual is not None:
            out = torch_npu.npu_add_rms_norm(x, residual, self.weight, self.variance_epsilon)
            return out[0], out[2]
        return torch_npu.npu_rms_norm(x, self.weight, self.variance_epsilon)[0]
