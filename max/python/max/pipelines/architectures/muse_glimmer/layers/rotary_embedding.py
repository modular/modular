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

"""Muse Glimmer NoPE rotary embedding for the ModuleV3 API."""

from __future__ import annotations

from max.driver import CPU
from max.dtype import DType
from max.experimental.nn.common_layers.rotary_embedding import RotaryEmbedding
from max.experimental.tensor import Tensor


class NoPERotaryEmbedding(RotaryEmbedding):
    """Identity RoPE for the full-attention (NoPE) layers.

    Every inverse frequency is zero, so ``freqs_cis`` is all ``(1, 0)``
    pairs. Full layers then store K and V through the same fused
    ``rope_split_store_ragged`` kernel as sliding layers: ModuleV3 has no
    store-without-rope wrapper.
    """

    def _compute_inv_freqs(self) -> Tensor:
        return Tensor.zeros(
            [self.head_dim // 2], dtype=DType.float32, device=CPU()
        )
