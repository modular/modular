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
"""Input batching for unified Hy4 MTP."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from max.driver import Buffer
from max.dtype import DType
from max.nn.kv_cache import KVCacheInputs, MultiKVCacheParams
from max.pipelines.context import TextContext
from max.pipelines.lib.interfaces.batch_processor import (
    BatchProcessorRuntime,
    UnifiedSpecDecodeBatchProcessor,
)

from ..hy_v4.batch_processor import prefill_buffer_length
from ..hy_v4.model_config import HYV4Config

if TYPE_CHECKING:
    from .model import UnifiedMTPHYV4Inputs


class UnifiedMTPHYV4BatchProcessor(
    UnifiedSpecDecodeBatchProcessor["UnifiedMTPHYV4Inputs", HYV4Config]
):
    """Ragged single-device batching plus the host prefill buffer length, as
    ``HYV4BatchProcessor`` writes it; the pipeline fills ``draft_tokens``."""

    def __init__(
        self, config: HYV4Config, runtime: BatchProcessorRuntime
    ) -> None:
        super().__init__(config, runtime)
        assert isinstance(config.kv_params, MultiKVCacheParams)
        self._page_size = config.kv_params.page_size
        self._batch_context_length = Buffer.zeros(shape=[1], dtype=DType.int32)

    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[TextContext]],
        kv_cache_inputs: KVCacheInputs[Buffer, Buffer] | None = None,
        return_n_logits: int = 1,
    ) -> UnifiedMTPHYV4Inputs:
        self._batch_context_length.to_numpy()[0] = prefill_buffer_length(
            [ctx for batch in replica_batches for ctx in batch],
            self._page_size,
        )
        return super().prepare_initial_token_inputs(
            replica_batches, kv_cache_inputs, return_n_logits
        )

    def _make_inputs(
        self,
        *,
        tokens: Buffer,
        input_row_offsets: Buffer,
        return_n_logits: Buffer,
        kv_cache_inputs: KVCacheInputs[Buffer, Buffer] | None,
        seed: Buffer,
        structured_output: bool,
    ) -> UnifiedMTPHYV4Inputs:
        from .model import UnifiedMTPHYV4Inputs

        return UnifiedMTPHYV4Inputs(
            tokens=tokens,
            input_row_offsets=input_row_offsets,
            return_n_logits=return_n_logits,
            kv_cache_inputs=kv_cache_inputs,
            seed=seed,
            structured_output=structured_output,
            batch_context_length=self._batch_context_length,
            signal_buffers=list(self.runtime.signal_buffers),
        )
