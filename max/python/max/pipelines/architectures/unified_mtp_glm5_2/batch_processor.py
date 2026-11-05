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
"""Input batching for the unified MTP GLM-5.2 pipeline model."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

from max.driver import Buffer
from max.nn.kv_cache import KVCacheInputs, KVCacheParamInterface
from max.pipelines.architectures.deepseekV3.batch_processor import (
    DeepseekV3BatchProcessorBase,
)
from max.pipelines.speculative.spec_input_types import SpecDecodeInputTypeSpec

if TYPE_CHECKING:
    from .model import UnifiedMTPGlm5_2Inputs


class UnifiedMTPGlm5_2BatchProcessor(
    DeepseekV3BatchProcessorBase["UnifiedMTPGlm5_2Inputs"]
):
    """Ragged batching for unified MTP GLM-5.2.

    Extends :class:`DeepseekV3BatchProcessor` to return
    :class:`UnifiedMTPGlm5_2Inputs` which carries the extra ``draft_tokens``
    slot required for speculative decoding. The slot is left ``None`` here; the
    overlap pipeline fills it in after this call returns.
    """

    input_spec: SpecDecodeInputTypeSpec | None = None
    """The fused graph's signature, which the model binds when it builds the
    graph and each step's inputs pack by."""

    def max_dims(
        self, dims: Iterable[str], kv_params: KVCacheParamInterface
    ) -> dict[str, int]:
        """Bounds the dims of a decoder's inputs and of the spec-decode tail."""
        return self.runtime.spec_decode_max_dims(dims, kv_params)

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
    ) -> UnifiedMTPGlm5_2Inputs:
        from .model import UnifiedMTPGlm5_2Inputs

        return UnifiedMTPGlm5_2Inputs(
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
            spec=self.input_spec,
        )
