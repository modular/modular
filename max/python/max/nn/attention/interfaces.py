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
"""General interface for Attention."""

from __future__ import annotations

from abc import ABC, abstractmethod

from max.graph import BufferValue, TensorValue

from ..kv_cache import PagedCacheValues
from ..layer import Module


class DistributedAttentionImpl(Module, ABC):
    """A generalized Distributed attention interface."""

    @abstractmethod
    def __call__(
        self,
        layer_idx: TensorValue,
        x: list[TensorValue],
        signal_buffers: list[BufferValue],
        kv_collections: list[PagedCacheValues],
        freqs_cis: list[TensorValue],
        input_row_offsets: TensorValue,
    ) -> list[TensorValue]: ...
