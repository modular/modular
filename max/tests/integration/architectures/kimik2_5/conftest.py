# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Shared test fixtures for Kimi K2.5 tests."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TorchMLP2(nn.Module):
    """PyTorch reference for the MLP2 layer (non-gated MLP with gelu_tanh)."""

    def __init__(
        self, dim: tuple[int, int, int], has_bias: bool = True
    ) -> None:
        super().__init__()
        self.up_proj = nn.Linear(dim[0], dim[1], bias=has_bias)
        self.down_proj = nn.Linear(dim[1], dim[2], bias=has_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up_proj(x)
        x = F.gelu(x, approximate="tanh")
        return self.down_proj(x)


class TorchRope2DPosEmbRepeated(nn.Module):
    """Nearly verbatim copy of the Kimi-K2.5 reference ``Rope2DPosEmbRepeated``."""

    def __init__(
        self,
        dim: int,
        max_height: int,
        max_width: int,
        theta_base: float = 10000,
    ):
        super().__init__()
        self.dim = dim
        assert self.dim % 4 == 0, "dim must be divisible by 4"
        self.max_height = max_height
        self.max_width = max_width
        self.theta_base = theta_base

    def _precompute_freqs_cis(self, device: torch.device) -> torch.Tensor:
        N = self.max_height * self.max_width
        flat_pos = torch.arange(0, N).float().to(device)
        x_pos = flat_pos % self.max_width
        y_pos = flat_pos // self.max_width
        dim_range = (
            torch.arange(0, self.dim, 4)[: (self.dim // 4)].float().to(device)
        )  # C/4
        freqs = 1.0 / (self.theta_base ** (dim_range / self.dim))
        x_freqs = torch.outer(x_pos, freqs).float()  # N, C/4
        y_freqs = torch.outer(y_pos, freqs).float()  # N, C/4
        x_cis = torch.polar(torch.ones_like(x_freqs), x_freqs)  # N, C/4
        y_cis = torch.polar(torch.ones_like(y_freqs), y_freqs)  # N, C/4
        # N, C/4, 2
        freqs_cis = torch.cat(
            [x_cis.unsqueeze(dim=-1), y_cis.unsqueeze(dim=-1)], dim=-1
        )
        # max_height, max_width, C/2
        freqs_cis = freqs_cis.reshape(self.max_height, self.max_width, -1)
        return freqs_cis

    def get_freqs_cis(
        self, grid_thws: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        if not hasattr(self, "freqs_cis"):
            self.register_buffer(
                "freqs_cis",
                self._precompute_freqs_cis(device),
                persistent=False,
            )

        shapes = grid_thws.tolist()
        assert all(
            1 <= h <= self.max_height and 1 <= w <= self.max_width
            for t, h, w in shapes
        ), (
            shapes,
            self.max_height,
            self.max_width,
        )
        freqs_cis = torch.cat(
            [
                self.freqs_cis[:h, :w].reshape(-1, self.dim // 2).repeat(t, 1)
                for t, h, w in shapes
            ],
            dim=0,
        )
        return freqs_cis
