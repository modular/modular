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

"""Standalone FirstBlockCache module for diffusion pipelines.

FirstBlockCache (FBCache) skips redundant transformer blocks during
denoising by comparing the first-block residual between consecutive
steps.  When the residual is sufficiently similar, the cached output
from the previous step is reused instead of running the remaining
transformer blocks.
"""

from __future__ import annotations

from dataclasses import dataclass

from max._core.driver import Device
from max.driver import Buffer
from max.dtype import DType
from max.experimental.tensor import Tensor


@dataclass
class FirstBlockCacheState:
    """Per-request mutable state for FirstBlockCache.

    Allocated fresh for each denoising request via
    ``FirstBlockCache.create_state()``.
    """

    prev_residual: Tensor | None = None
    """First-block output residual from the previous step."""

    prev_output: Tensor | None = None
    """Full transformer output from the previous step."""


class FirstBlockCache:
    """Standalone FirstBlockCache module.

    Provides state allocation for FBCache.  The conditional execution
    helpers (``can_use_fbcache``, ``fbcache_conditional_execution``)
    remain in ``cache_mixin.py`` since they are used directly inside
    transformer ``_forward_fbcache`` methods.
    """

    def __init__(self, dtype: DType, device: Device) -> None:
        self.dtype = dtype
        self.device = device

    def create_state(
        self,
        batch_size: int,
        seq_len: int,
        residual_dim: int,
        output_dim: int,
    ) -> FirstBlockCacheState:
        """Allocate fresh per-request FirstBlockCache state tensors."""

        def _zeros(shape: tuple[int, ...]) -> Tensor:
            return Tensor(
                storage=Buffer.zeros(shape, self.dtype, device=self.device)
            )

        return FirstBlockCacheState(
            prev_residual=_zeros((batch_size, seq_len, residual_dim)),
            prev_output=_zeros((batch_size, seq_len, output_dim)),
        )
