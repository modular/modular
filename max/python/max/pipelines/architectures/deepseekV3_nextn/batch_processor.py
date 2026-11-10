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
"""Input batching for DeepseekV3 NextN pipeline models."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from max.driver import Buffer
from max.nn.kv_cache import KVCacheInputs
from max.pipelines.architectures.deepseekV3.batch_processor import (
    DeepseekV3BatchProcessorBase,
)
from max.pipelines.context import TextContext

if TYPE_CHECKING:
    from .model import DeepseekV3NextNInputs


class DeepseekV3NextNBatchProcessor(
    DeepseekV3BatchProcessorBase["DeepseekV3NextNInputs"]
):
    """Ragged batching for the DeepseekV3 NextN (MTP draft) model.

    Inherits the base DeepseekV3 batching logic but overrides the
    ``batch_context_lengths`` update to use raw active-token lengths (without
    page-size alignment), matching the NextN model's simpler context tracking.

    The ``hidden_states`` field on :class:`DeepseekV3NextNInputs` is left
    ``None`` on creation; the EAGLE pipeline sets it via
    ``model_inputs.update(hidden_states=...)`` before calling ``execute()``.
    """

    def _update_batch_context_lengths(
        self,
        replica_batches: Sequence[Sequence[TextContext]],
    ) -> None:
        """Update per-device batch context lengths using raw active lengths."""
        dp = self.runtime.pipeline_config.model.data_parallel_degree

        for i, batch in enumerate(replica_batches):
            curr_length = sum(ctx.tokens.active_length for ctx in batch)
            self._batch_context_lengths[i][0] = curr_length

        if dp != len(self.runtime.devices):
            assert dp == 1
            for dev_idx in range(1, len(self.runtime.devices)):
                self._batch_context_lengths[dev_idx][0] = (
                    self._batch_context_lengths[0][0].item()
                )

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
    ) -> DeepseekV3NextNInputs:
        from .model import DeepseekV3NextNInputs

        return DeepseekV3NextNInputs(
            tokens=tokens,
            input_row_offsets=input_row_offsets,
            host_input_row_offsets=host_input_row_offsets,
            batch_context_lengths=batch_context_lengths,
            signal_buffers=signal_buffers,
            kv_cache_inputs=kv_cache_inputs,
            return_n_logits=return_n_logits,
            data_parallel_splits=data_parallel_splits,
            ep_inputs=ep_inputs,
        )
