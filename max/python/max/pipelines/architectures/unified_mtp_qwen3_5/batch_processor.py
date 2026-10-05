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
"""Input batching for the fused Qwen3.5 MTP and DFlash2 graphs."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
from max.driver import Buffer
from max.dtype import DType
from max.nn.kv_cache import KVCacheInputs, RecurrentStateInputsPerDevice
from max.pipelines.context import TextContext
from max.pipelines.kv_cache.paged_kv_cache.cache_manager import (
    prompt_tokens_for_context,
)
from max.pipelines.lib.interfaces.arch_config import ArchConfig
from max.pipelines.lib.interfaces.batch_processor import BatchProcessorRuntime
from max.pipelines.lib.utils import compute_data_parallel_splits
from max.support.algorithm import flatten2d
from typing_extensions import override

from ..qwen3_5.batch_processor import (
    Qwen3_5BatchProcessor,
    context_position_rows,
)
from ..qwen3_5.state_cache import STATE_CACHE_KEY
from .inputs import UnifiedMTPQwen3_5Inputs


def merged_position_rows(
    contexts: Sequence[TextContext],
) -> npt.NDArray[np.int64]:
    """Returns the ``[3, merged_total_seq_len]`` positions the graph declares.

    Each request's window is its real tokens followed by its ``K`` drafts, in
    the layout
    :class:`~max.pipelines.speculative.ragged_token_merger.RaggedTokenMerger`
    produces. Draft positions continue from the request's last real
    position, so they stay correct for a request whose positions carry an
    M-RoPE correction.

    Args:
        contexts: The batch, in the order the merged tokens are built in.

    Returns:
        Three rows of int64: temporal, height and width.
    """
    rows: list[npt.NDArray[np.int64]] = []
    for ctx in contexts:
        real = context_position_rows(ctx)
        num_drafts = len(ctx.spec_decoding_state.draft_tokens_to_verify)
        if num_drafts:
            ramp = np.arange(1, num_drafts + 1, dtype=np.int64)
            real = np.concatenate([real, real[:, -1:] + ramp], axis=1)
        assert real.shape[1] == prompt_tokens_for_context(ctx), (
            f"{ctx.request_id}: {real.shape[1]} positions for a"
            f" {prompt_tokens_for_context(ctx)}-token merged window"
        )
        rows.append(real)
    return np.concatenate(rows, axis=1).astype(np.int64)


class UnifiedMTPQwen3_5BatchProcessor(Qwen3_5BatchProcessor):
    """Builds a batch for either fused Qwen3.5 speculative graph.

    The MTP and DFlash2 graphs take the same inputs, and only MTP can declare
    M-RoPE positions. The cache's attention children go in the KV slice and
    its state child, the verify ring included, goes in the trailing tail.
    Image prompts are rejected because neither graph has a vision encoder.
    """

    def __init__(
        self, config: ArchConfig, runtime: BatchProcessorRuntime
    ) -> None:
        super().__init__(config, runtime)
        # Required by the signature but unused by this graph.
        self._batch_context_lengths = [
            Buffer.zeros(shape=[1], dtype=DType.int32)
            for _ in range(len(runtime.devices))
        ]

    def _reject_image_prompts(self, contexts: Sequence[TextContext]) -> None:
        """Raises if any request carries images.

        Checks for images being present rather than
        ``needs_vision_encoding``, which is false for later prefill chunks.
        """
        # TODO(kevinbi): this error ends the model worker instead of failing
        # one request, because `SchedulerResult.failed` does not cover batch
        # preparation.
        if any(getattr(ctx, "images", None) for ctx in contexts):
            raise ValueError(
                "Speculative Qwen3.5 cannot serve image prompts because its"
                " fused graph has no vision encoder. Drop"
                " --speculative-method to serve images on Qwen3_5."
            )

    def _state_tail(
        self, kv_cache_inputs: KVCacheInputs[Buffer, Buffer]
    ) -> tuple[
        KVCacheInputs[Buffer, Buffer],
        tuple[RecurrentStateInputsPerDevice[Buffer, Buffer], ...],
    ]:
        """Splits the cache into its attention children and state leaves.

        Returns the attention children in declaration order, and the state
        child's per-device leaves.
        """
        assert isinstance(kv_cache_inputs, Mapping), (
            f"expected a cache tree, got {type(kv_cache_inputs).__name__}"
        )
        state = kv_cache_inputs.get(STATE_CACHE_KEY)
        assert state is not None, "no recurrent state child in the cache"
        # `tree.leaves` flattens a plain dict in sorted key order, which
        # would put "draft" before "target".
        attention: OrderedDict[str, Any] = OrderedDict(
            (key, child)
            for key, child in kv_cache_inputs.items()
            if key != STATE_CACHE_KEY
        )
        assert isinstance(state, (list, tuple))
        per_device: list[RecurrentStateInputsPerDevice[Buffer, Buffer]] = []
        for leaves in state:
            assert isinstance(leaves, RecurrentStateInputsPerDevice)
            per_device.append(leaves)
        return attention, tuple(per_device)

    @override
    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[TextContext]],
        kv_cache_inputs: KVCacheInputs[Buffer, Buffer] | None = None,
        return_n_logits: int = 1,
    ) -> UnifiedMTPQwen3_5Inputs:
        contexts = flatten2d(replica_batches)
        self._reject_image_prompts(contexts)

        assert kv_cache_inputs is not None
        attention, state = self._state_tail(kv_cache_inputs)

        tokens, row_offsets, host_row_offsets = self._stage_ragged_token_inputs(
            contexts
        )

        # ``linear_state_regions`` declares conv, then recurrent, then the
        # ring. The graph takes each leaf's pools and rows region-major,
        # device-minor.
        conv = [per_device.leaves[0] for per_device in state]
        recurrent = [per_device.leaves[1] for per_device in state]
        rings = [per_device.leaves[2] for per_device in state]

        return UnifiedMTPQwen3_5Inputs(
            tokens=tokens,
            input_row_offsets=row_offsets,
            # Always passed, since this graph always declares them.
            host_input_row_offsets=host_row_offsets,
            return_n_logits=Buffer.from_numpy(
                np.array([return_n_logits], dtype=np.int64)
            ),
            data_parallel_splits=Buffer.from_numpy(
                compute_data_parallel_splits(replica_batches)
            ),
            signal_buffers=list(self.runtime.signal_buffers),
            batch_context_lengths=list(self._batch_context_lengths),
            kv_cache_inputs=attention,
            live_conv_pools=[leaf.pool for leaf in conv],
            live_recurrent_pools=[leaf.pool for leaf in recurrent],
            live_conv_row_ids=[leaf.live_row_ids for leaf in conv],
            live_recurrent_row_ids=[leaf.live_row_ids for leaf in recurrent],
            ring_pools=[leaf.pool for leaf in rings],
            ring_row_ids=[leaf.live_row_ids for leaf in rings],
            position_ids=(
                Buffer.from_numpy(merged_position_rows(contexts)).to(
                    self.runtime.devices[0]
                )
                if self.mrope_enabled
                else None
            ),
            draft_tokens=None,
            structured_output=self.runtime.pipeline_config.needs_bitmask_constraints,
        )
