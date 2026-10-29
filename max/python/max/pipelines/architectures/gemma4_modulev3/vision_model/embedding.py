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

"""Gemma4 vision patch embedder for the ModuleV3 API."""

from __future__ import annotations

from max import functional as F
from max.dtype import DType
from max.nn import Module
from max.nn.linear import Linear
from max.pipelines.architectures.gemma4.model_config import (
    Gemma4ForConditionalGenerationConfig,
)
from max.tensor import Tensor


class Gemma4VisionPatchEmbedder(Module[..., Tensor]):
    """Turns raw pixels into embedded patches.

    Each pixel is rescaled from ``[0, 1]`` to ``[-1, 1]``, projected
    linearly, and given its 2D position embedding. This is the ModuleV3 port
    of the graph architecture's ``Gemma4VisionPatchEmbedder``.
    """

    def __init__(self, config: Gemma4ForConditionalGenerationConfig) -> None:
        super().__init__()
        vision_cfg = config.vision_config
        assert vision_cfg is not None
        self.dtype = config.unquantized_dtype
        self.hidden_size = vision_cfg.hidden_size
        self.position_embedding_size = vision_cfg.position_embedding_size

        self.input_proj = Linear(
            3 * vision_cfg.patch_size**2, self.hidden_size, bias=False
        )
        self.position_embedding_table = Tensor.zeros(
            [2, self.position_embedding_size, self.hidden_size]
        )

    def forward(
        self, patches_flat: Tensor, pixel_position_ids: Tensor
    ) -> Tensor:
        """Scales the pixels, projects them and adds their positions."""
        patches = patches_flat.cast(self.dtype) * 2.0 - 1.0
        hidden = self.input_proj(patches)

        table_flat = self.position_embedding_table.reshape(
            (2 * self.position_embedding_size, self.hidden_size)
        )
        x_ids = pixel_position_ids[:, 0].cast(DType.int64)
        y_ids = (
            pixel_position_ids[:, 1].cast(DType.int64)
            + self.position_embedding_size
        )
        position_emb = F.gather(table_flat, x_ids, axis=0) + F.gather(
            table_flat, y_ids, axis=0
        )
        return hidden + position_emb
