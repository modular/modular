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
"""Kimi K3 image tower, derived from K2.5 with K3 norms and projector."""

from __future__ import annotations

from collections import OrderedDict
from typing import ClassVar

from max import functional as F
from max.driver import CPU
from max.dtype import DType
from max.graph import DeviceRef, TensorType
from max.nn import Module
from max.nn._graph_ops import tpool_patch_merger as _tpool_patch_merger
from max.nn.legacy.graph_groups import VisionHostMax, VisionRows, VisionSegments
from max.tensor import Tensor, default_dtype
from max.tensor.sharding import AxisAssignment, replicated_rows
from max.tensor.sharding.types import TensorLayout

from ...model_config import KimiK3Config
from ...vision_config import KimiK3VisionConfig
from ...vision_preprocessing import vision_metadata, vision_preprocess_spec
from .encoder import Encoder
from .patch_embedding import PatchEmbedding
from .patch_merger import PatchMergerMLP


def _merge_rule(
    input: TensorLayout,
    grid_thws: TensorLayout,
    kH: int,
    kW: int,
    max_h: int | TensorLayout,
    max_w: int | TensorLayout,
) -> list[AxisAssignment]:
    return replicated_rows(input, grid_thws, max_h, max_w)


tpool_patch_merger = F.functional(_tpool_patch_merger, rule=_merge_rule)


class Transformer(Module[[Tensor, Tensor, Tensor, Tensor, Tensor], Tensor]):
    """Encodes image patches and projects merged groups to decoder width."""

    def __init__(self, config: KimiK3VisionConfig) -> None:
        self.vision_config = config
        with default_dtype(config.dtype):
            self.patch_embed = PatchEmbedding(config)
            self.encoder = Encoder(config)
            self.patch_merger = PatchMergerMLP(config)

    def forward(
        self,
        pixel_values: Tensor,
        grid_thws: Tensor,
        input_row_offsets: Tensor,
        max_seq_len: Tensor,
        position_ids: Tensor,
    ) -> Tensor:
        """Returns the merged image embeddings in grid-table order."""
        x = self.patch_embed(pixel_values, grid_thws)
        x = self.encoder(x, input_row_offsets, max_seq_len, position_ids)
        local_grid = grid_thws.local_shards[0]
        max_h = (
            F.max(local_grid[:, 1], axis=0)
            .cast(DType.int32)
            .reshape([])
            .to(CPU())
        )
        max_w = (
            F.max(local_grid[:, 2], axis=0)
            .cast(DType.int32)
            .reshape([])
            .to(CPU())
        )
        kH, kW = self.vision_config.merge_kernel_size
        x = tpool_patch_merger(
            x, grid_thws, kH=kH, kW=kW, max_h=max_h, max_w=max_w
        )
        merge_k = kH * kW
        x = F.rebind(x, [(x.shape[0] // merge_k) * merge_k, x.shape[1]])
        x = x.reshape([x.shape[0] // merge_k, merge_k, x.shape[1]])
        return self.patch_merger(x)


class KimiK3VisionEncoder(Transformer):
    """Registers the single-device still-image tower for the VISION role."""

    graph_name: ClassVar[str] = "kimi_k3_vision"
    vision_metadata = staticmethod(vision_metadata)
    vision_preprocess_spec = staticmethod(vision_preprocess_spec)

    @classmethod
    def is_enabled(cls, config: KimiK3Config) -> bool:
        """Skips the encoder for checkpoints without a vision tower."""
        return config.vision_config is not None

    def __init__(self, config: KimiK3Config) -> None:
        vision_config = config.vision_config
        assert vision_config is not None
        super().__init__(vision_config)
        self.embedding_dtype = DType.bfloat16
        self.embedding_dim = vision_config.text_hidden_size

    def input_types(self) -> OrderedDict[str, object]:
        """Groups the five inputs by their serving batch packing rules."""
        config = self.vision_config
        device = config.devices[0]
        return OrderedDict(
            patches=VisionRows(
                pixel_values=TensorType(
                    config.dtype,
                    [
                        "vision_rows",
                        config.in_channels,
                        config.patch_size,
                        config.patch_size,
                    ],
                    device,
                ),
                position_ids=TensorType(DType.int64, ["vision_rows"], device),
            ),
            grids=VisionRows(
                grid_thws=TensorType(DType.int64, ["n_images", 3], device)
            ),
            segments=VisionSegments(
                input_row_offsets=TensorType(
                    DType.uint32, ["n_segments"], device
                )
            ),
            segment_bound=VisionHostMax(
                max_seq_len=TensorType(DType.uint32, [1], DeviceRef.CPU())
            ),
        )
