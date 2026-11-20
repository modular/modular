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
"""Bidirectional MHSA for the SigLIP2 NaFlex tower.

HF reference: modeling_siglip2.py::Siglip2Attention, run on the
create_bidirectional_mask the vision model builds from
`pixel_attention_mask`. Head layout: 16 heads, head_dim 72 (hidden 1152),
q/k/v/out projections ALL WITH BIAS, no q/k norm, no rope, no GQA — the
plain SigLIP attention the lfm3-vl tower's GQA machinery deliberately
does not match.

The reference attends over a PADDED row canvas with an additive mask;
the port attends over UNPADDED patch rows with each tile its own
attention row (`cu_seqlens`). That is exact: the mask only ever removes
pad positions, and each padded canvas row is one tile — tiles never
attend to each other in the reference either.

`flash_attention_ragged_gpu` (cache-less ragged kernel, NULL_MASK) is
the same kernel the lfm3-vl tower uses, with none of its q/k-norm or
rope prelude.
"""

from __future__ import annotations

from max.nn import Module
from max.nn.attention_mask import MHAMaskVariant
from max.nn.kernels import flash_attention_ragged_gpu
from max.nn.linear import Linear
from max.tensor import Tensor

from ..model_config import Lfm2VlConfig


class Lfm2VlVisionAttention(Module[..., Tensor]):
    """Tile-local bidirectional MHSA with biases."""

    def __init__(self, config: Lfm2VlConfig) -> None:
        super().__init__()
        self.num_heads = config.vision_num_attention_heads
        self.hidden_size = config.vision_hidden_size
        self.head_dim = config.vision_head_dim
        self.scale = self.head_dim**-0.5
        self.q_proj = Linear(
            in_dim=self.hidden_size,
            out_dim=self.hidden_size,
            bias=True,
        )
        self.k_proj = Linear(
            in_dim=self.hidden_size,
            out_dim=self.hidden_size,
            bias=True,
        )
        self.v_proj = Linear(
            in_dim=self.hidden_size,
            out_dim=self.hidden_size,
            bias=True,
        )
        self.out_proj = Linear(
            in_dim=self.hidden_size,
            out_dim=self.hidden_size,
            bias=True,
        )

    def forward(
        self,
        x: Tensor,
        tile_offsets: Tensor,
        max_tile_len: Tensor,
    ) -> Tensor:
        """Tile-local bidirectional MHSA over the ragged patch rows."""
        tokens = x.shape[0]
        heads = self.num_heads
        head_dim = self.head_dim

        q = self.q_proj(x).reshape([tokens, heads, head_dim])
        k = self.k_proj(x).reshape([tokens, heads, head_dim])
        v = self.v_proj(x).reshape([tokens, heads, head_dim])

        attended = flash_attention_ragged_gpu(
            q,
            k,
            v,
            input_row_offsets=tile_offsets,
            max_seq_len=max_tile_len,
            mask_variant=MHAMaskVariant.NULL_MASK,
            scale=self.scale,
        )
        return self.out_proj(attended.reshape([tokens, self.hidden_size]))
