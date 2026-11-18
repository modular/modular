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
"""The LFM2-VL multi-modal projector: the 2x2 fold and two linears.

HF reference: modeling_lfm2_vl.py::Lfm2VlMultiModalProjector —

    h = pixel_unshuffle(image_features)   # fold each 2x2 patch block
    h = linear_1(h); h = gelu(h); h = linear_2(h)   # -> text hidden

`pixel_unshuffle` on a [1, feature_h, feature_w, C] tile reshapes and
permutes down to [1, feature_h // 2, feature_w // 2, C * 4], and the
flattened result's row order is raster over (block_h, block_w) with each
block's four patches ordered (jh, jw) — [top-left, top-right,
bottom-left, bottom-right] — and channels innermost. The four patches of
a block are NOT consecutive rows of the tower output (they span two
patch rows), so the fold is a GATHER over the tower's rows: the batch
processor precomputes, per image token, the four patch row indices
(`block_gather`, [num_image_tokens, 4]) and the graph reshapes the
gathered [tokens, 4, C] to the [tokens, 4C] the linear eats.

`projector_use_layernorm=false` for the shipped checkpoint, so there is
NO LayerNorm before linear_1; the config gate refuses a checkpoint that
wants one. `gelu` here is
HF's ACT2FN["gelu"], the exact erf form — MAX's `F.gelu` defaults to
the same (`approximate="none"`), not the tower MLP's tanh form.
"""

from __future__ import annotations

from max import functional as F
from max.nn import Module
from max.nn.linear import Linear
from max.tensor import Tensor

from ..model_config import Lfm2VlConfig


class Lfm2VlMultiModalProjector(Module[..., Tensor]):
    """2x2-block gather, then linear_1 -> gelu -> linear_2 (with bias)."""

    def __init__(self, config: Lfm2VlConfig) -> None:
        super().__init__()
        self.fold = config.vision_hidden_size * config.downsample_factor**2
        self.linear_1 = Linear(
            in_dim=self.fold,
            out_dim=config.projector_hidden_size,
            bias=config.projector_bias,
        )
        self.linear_2 = Linear(
            in_dim=config.projector_hidden_size,
            out_dim=config.hidden_size,
            bias=config.projector_bias,
        )

    def forward(self, image_features: Tensor, block_gather: Tensor) -> Tensor:
        """Gather each image token's 2x2 fold and project to text width."""
        tokens = block_gather.shape[0]
        # [tokens, 4] patch indices -> [tokens * 4] -> gather -> [tokens,
        # 4, C] -> the (jh, jw, c) layout linear_1's weight expects.
        flats = block_gather.reshape([tokens * 4])
        blocks = F.gather(image_features, flats, axis=0).reshape(
            [tokens, self.fold]
        )
        return self.linear_2(F.gelu(self.linear_1(blocks)))
