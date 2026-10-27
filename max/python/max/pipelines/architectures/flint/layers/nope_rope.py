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

"""An identity rope variant for layers with no positional encoding.

Keeping the rope interface lets a NoPE layer share the fused store path.
"""

from __future__ import annotations

from typing import Any

from max.experimental.nn.common_layers.rotary_embedding import (
    RotaryEmbedding,
)
from max.experimental.tensor import Tensor


class NopeRotaryEmbedding(RotaryEmbedding):
    """A rope variant whose inverse frequencies are all zero."""

    def _compute_inv_freqs(self) -> Tensor:
        return super()._compute_inv_freqs() * 0.0


# Older ports name the same class IdentityRotaryEmbedding.
IdentityRotaryEmbedding = NopeRotaryEmbedding


def no_rope_layers(
    hf_cfg: Any,
    n_layers: int | None = None,
    default_interval: int = 4,
) -> list[int]:
    """Returns the per-layer rope flags, where 1 means the layer uses rope."""
    n = n_layers if n_layers is not None else hf_cfg.num_hidden_layers
    explicit = getattr(hf_cfg, "no_rope_layers", None)
    if explicit:
        return [int(x) for x in explicit[:n]]
    interval = int(
        getattr(hf_cfg, "no_rope_layer_interval", default_interval)
        or default_interval
    )
    return [int((i + 1) % interval != 0) for i in range(n)]
