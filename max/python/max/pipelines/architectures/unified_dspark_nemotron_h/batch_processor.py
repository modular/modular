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
"""Input batching for the unified DSpark Nemotron-H pipeline model."""

from __future__ import annotations

from typing import TYPE_CHECKING

from max.driver import Buffer
from max.nn.kv_cache import KVCacheInputs
from max.pipelines.lib.interfaces.batch_processor import (
    UnifiedSpecDecodeBatchProcessor,
)

if TYPE_CHECKING:
    from .model import UnifiedDSparkNemotronHInputs


class UnifiedDSparkNemotronHBatchProcessor(
    UnifiedSpecDecodeBatchProcessor["UnifiedDSparkNemotronHInputs"]
):
    """Ragged batching with persistent buffers and a seed, for DSpark.

    The graph is traced by ``Module.compile``, so the signal buffers of a
    multi-device mesh trail every other input.
    """

    def _make_inputs(
        self,
        *,
        tokens: Buffer,
        input_row_offsets: Buffer,
        return_n_logits: Buffer,
        kv_cache_inputs: KVCacheInputs[Buffer, Buffer] | None,
        seed: Buffer,
        structured_output: bool,
    ) -> UnifiedDSparkNemotronHInputs:
        from .model import UnifiedDSparkNemotronHInputs

        return UnifiedDSparkNemotronHInputs(
            tokens=tokens,
            input_row_offsets=input_row_offsets,
            return_n_logits=return_n_logits,
            signal_buffers=list(self.runtime.signal_buffers),
            kv_cache_inputs=kv_cache_inputs,
            seed=seed,
            structured_output=structured_output,
        )
