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
"""Input batching for the unified MTP DeepseekV3 pipeline model."""

from __future__ import annotations

from typing import TYPE_CHECKING

from max.driver import Buffer
from max.nn.kv_cache import KVCacheInputs
from max.pipelines.architectures.deepseekV3.batch_processor import (
    DeepseekV3BatchProcessorBase,
)

if TYPE_CHECKING:
    from .model import UnifiedMTPDeepseekV3Inputs


class UnifiedMTPDeepseekV3BatchProcessor(
    DeepseekV3BatchProcessorBase["UnifiedMTPDeepseekV3Inputs"]
):
    """Ragged batching for unified MTP DeepseekV3.

    Extends :class:`DeepseekV3BatchProcessor` to return
    :class:`UnifiedMTPDeepseekV3Inputs` which carries the extra ``draft_tokens``
    slot required for speculative decoding.  The slot is left ``None`` here;
    the overlap pipeline fills it in after this call returns.
    """

    def _make_mla_inputs(
        self,
        *,
        tokens: Buffer,
        input_row_offsets: Buffer,
        host_input_row_offsets: Buffer,
        batch_context_lengths: list[Buffer],
        signal_buffers: list[Buffer],
        kv_cache_inputs: KVCacheInputs[Buffer, Buffer] | None,
        return_n_logits: Buffer,
        data_parallel_splits: Buffer,
        ep_inputs: tuple[Buffer, ...],
    ) -> UnifiedMTPDeepseekV3Inputs:
        from .model import UnifiedMTPDeepseekV3Inputs

        return UnifiedMTPDeepseekV3Inputs(
            tokens=tokens,
            input_row_offsets=input_row_offsets,
            host_input_row_offsets=host_input_row_offsets,
            batch_context_lengths=batch_context_lengths,
            signal_buffers=signal_buffers,
            kv_cache_inputs=kv_cache_inputs,
            return_n_logits=return_n_logits,
            data_parallel_splits=data_parallel_splits,
            ep_inputs=ep_inputs,
            draft_tokens=None,
            structured_output=self.runtime.pipeline_config.needs_bitmask_constraints,
        )
