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
"""Kimi K3 patch embedding, derived from K2.5 with bilinear positions."""

from __future__ import annotations

import numpy as np
from max import functional as F
from max.dtype import DType
from max.nn import Module
from max.nn._graph_ops import learnable_2d_interp_pos_emb as _position_embedding
from max.nn.conv import Conv2d
from max.tensor import Tensor
from max.tensor.sharding import AxisAssignment, replicated_rows
from max.tensor.sharding.types import TensorLayout

from ...vision_config import KimiK3VisionConfig


def _position_rule(
    x: TensorLayout,
    weight: TensorLayout,
    grid_thws: TensorLayout,
    time_weight: TensorLayout,
    *extras: object,
) -> list[AxisAssignment]:
    return replicated_rows(x, weight, grid_thws, time_weight)


position_embedding = F.functional(_position_embedding, rule=_position_rule)


class Learnable2DInterpPosEmbDividedFixed(Module[[Tensor, Tensor], Tensor]):
    """Adds a bilinearly interpolated positional grid to image patches."""

    def __init__(self, config: KimiK3VisionConfig) -> None:
        self.weight = Tensor.zeros(
            [
                config.init_pos_emb_height,
                config.init_pos_emb_width,
                config.vt_hidden_size,
            ],
            dtype=config.dtype,
        )
        half = config.vt_hidden_size // 2
        omega = 1.0 / (10000.0 ** (np.arange(half, dtype=np.float32) / half))
        angles = np.outer(
            np.arange(config.init_pos_emb_time, dtype=np.float32), omega
        )
        self._time_weight = np.concatenate(
            [np.sin(angles), np.cos(angles)], axis=1
        )

    def forward(self, x: Tensor, grid_thws: Tensor) -> Tensor:
        """Adds the checkpoint's bilinear image position embeddings."""
        return position_embedding(
            x=x,
            weight=self.weight,
            grid_thws=grid_thws,
            time_weight=F.constant(
                self._time_weight, dtype=DType.float32, device=x.device
            ),
            interpolation_mode="bilinear",
        )


class PatchEmbedding(Module[[Tensor, Tensor], Tensor]):
    """Projects packed image patches and adds their positional embeddings."""

    def __init__(self, config: KimiK3VisionConfig) -> None:
        self.hidden_size = config.vt_hidden_size
        self.proj = Conv2d(
            in_channels=config.in_channels,
            out_channels=config.vt_hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            dtype=config.dtype,
            has_bias=False,
            permute=True,
        )
        self.pos_emb = Learnable2DInterpPosEmbDividedFixed(config)

    def forward(self, pixel_values: Tensor, grid_thws: Tensor) -> Tensor:
        """Projects each patch from its NCHW pixels."""
        x = self.proj(pixel_values.cast(self.proj.weight.dtype))
        return self.pos_emb(
            x.reshape([x.shape[0], self.hidden_size]), grid_thws
        )
