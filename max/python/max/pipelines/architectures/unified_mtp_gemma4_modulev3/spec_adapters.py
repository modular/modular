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

"""Speculative-decoding adapters for a ModuleV3 Gemma4 target and draft.

A ModuleV3 port of ``unified_mtp_gemma4.spec_adapters``, as
:class:`TensorAdapter` s: the driver hands them single-device
:class:`Tensor` values and they call the traced target/draft modules
directly, rebuilding per-device lists as replicated distributed tensors.
"""

from __future__ import annotations

from collections.abc import Sequence

from max.driver import CPU
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn.common_layers.kv_cache import PagedCacheValues
from max.experimental.sharding import DeviceMapping, DeviceMesh
from max.experimental.tensor import Tensor
from max.graph import BufferType, DimLike, TensorType
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.speculative.driver import (
    CarryDimNames,
    DecodeKVSwap,
    DraftCache,
    DraftStepInput,
    Proposed,
    ReuseSpec,
    SequentialBatch,
    TensorAdapter,
)
from max.pipelines.speculative.spec_target import Verified

from ..gemma4_modulev3.gemma4 import Gemma4TextModel
from .gemma4_assistant_modulev3 import Gemma4AssistantModuleV3

__all__ = ["GLOBAL_KV", "Gemma4ModuleV3MTPProposer", "Gemma4ModuleV3Target"]

GLOBAL_KV = "full_attention"
"""Name the full-attention leaf rides under in ``passthrough_kv``.

The ModuleV3 module's graph signature names this same leaf, so both
adapters below address it identically."""


def _host_q_max_seq_len(host_offsets: Tensor) -> Tensor:
    """Longest per-request query in this call, for the cross-attention kernel."""
    lengths = host_offsets[1:] - host_offsets[:-1]
    return F.max(lengths, axis=0).cast(DType.uint32).broadcast_to([1])


class Gemma4ModuleV3Target(TensorAdapter):
    """The ModuleV3 Gemma4 target entry point and its output layout.

    Verification reads VARIABLE logits and every token's normalized hidden
    state, as the graph-API target does; the last-token logits the spec
    loop never reads are still computed (the target's graph computes them
    unconditionally).
    """

    def __init__(self, target: Gemma4TextModel) -> None:
        self.target = target
        # Belt-and-braces with the instance attribute the subclass's
        # forward sets: the traced module's modes are fixed at build time.
        target.return_logits = ReturnLogits.VARIABLE

    def verify(
        self, batch: SequentialBatch[Tensor, Tensor, Tensor]
    ) -> Verified[list[Tensor], Tensor]:
        n_devs = len(self.target.mesh.devices)
        replicated = DeviceMapping.replicated(self.target.mesh)
        sliding_kv = PagedCacheValues.from_upstream(
            batch.kv_collections, replicated
        )
        global_kv = PagedCacheValues.from_upstream(
            batch.passthrough_kv[GLOBAL_KV], replicated
        )
        # Vision embeddings scatter into the target at
        # ``vision_scatter_indices``. Images only appear during prefill,
        # where K = 0; during decode both lists are zero-row and this is a
        # no-op.
        outputs = self.target(
            batch.merged_tokens,
            sliding_kv,
            global_kv,
            batch.return_n_logits,
            # The target broadcasts the merged tokens and offsets itself.
            batch.merged_offsets,
            batch.vision_embeddings[0],
            batch.vision_scatter_indices[0],
        )
        # VARIABLE logits + ALL-normalized hidden states ->
        # (last_logits, logits, offsets, hs_0..hs_{n-1}).
        # The logits are replicated after the vocab allgather; the speculative
        # framework takes the single-device tensor on the first device.
        return Verified(
            logits=outputs[1].local_shards[0],
            hidden=list(outputs[3 : 3 + n_devs]),
        )

    def ep_input_types(self) -> Sequence[TensorType | BufferType]:
        return ()


class Gemma4ModuleV3MTPProposer(TensorAdapter):
    """The Gemma4 assistant: Q-only cross-attention into the target's caches.

    A ModuleV3 port of ``Gemma4MTPProposer``: same swaps, same carry
    names, same output layout; only the draft call takes ``Tensor`` values
    and passes the modes as forward arguments rather than mutating the
    traced module.
    """

    decode_swaps: tuple[DecodeKVSwap, ...] = (
        DecodeKVSwap.MAX_PROMPT_LENGTH_ONE,
    )
    passthrough_decode_swaps: tuple[str, ...] = (GLOBAL_KV,)
    draft_cache = DraftCache.TARGET
    reuse: ReuseSpec | None = None
    split_prefix = "mtp"
    carry_dim_names = CarryDimNames(prefix="mtp_step")
    step_hidden_mode = ReturnHiddenStates.LAST_PER_DEVICE
    uses_thinking_phase = True
    supports_zero_draft_rows = True

    def __init__(
        self,
        draft: Gemma4AssistantModuleV3,
        hidden_dim: DimLike,
    ) -> None:
        self.draft = draft
        self.hidden_dim = hidden_dim
        self.mesh: DeviceMesh = draft.mesh

    def prefill(
        self,
        batch: SequentialBatch[Tensor, Tensor, Tensor],
        tokens: Tensor,
        target_hidden: list[Tensor],
    ) -> Proposed[Tensor]:
        outputs = self.draft(
            tokens,
            target_hidden,
            PagedCacheValues.from_upstream(
                batch.draft_kv_collections,
                DeviceMapping.replicated(self.mesh),
            ),
            PagedCacheValues.from_upstream(
                batch.passthrough_kv[GLOBAL_KV],
                DeviceMapping.replicated(self.mesh),
            ),
            batch.return_n_logits,
            batch.query_offsets_per_dev,
            batch.merged_offsets_per_dev,
            _host_q_max_seq_len(batch.dist.host_query_offsets),
            None,
            return_logits=ReturnLogits.VARIABLE,
            return_hidden_states=ReturnHiddenStates.ALL,
        )
        n_devs = len(self.mesh.devices)
        # (last_logits, logits, offsets, hs_0..hs_{n-1}).
        return Proposed(
            logits=outputs[1].local_shards[0],
            hidden=list(outputs[3 : 3 + n_devs]),
        )

    def step(
        self,
        batch: SequentialBatch[Tensor, Tensor, Tensor],
        draft_input: DraftStepInput[Tensor],
        index: int,
    ) -> Proposed[Tensor]:
        del index
        outputs = self.draft(
            draft_input.tokens,
            draft_input.hidden,
            PagedCacheValues.from_upstream(
                batch.draft_kv_collections,
                DeviceMapping.replicated(self.mesh),
            ),
            PagedCacheValues.from_upstream(
                batch.passthrough_kv[GLOBAL_KV],
                DeviceMapping.replicated(self.mesh),
            ),
            batch.return_n_logits,
            batch.query_offsets_per_dev,
            # The cache still spans the merged verify window, so these stay
            # the merged offsets.
            batch.merged_offsets_per_dev,
            F.constant(1, DType.uint32, device=CPU()).broadcast_to([1]),
            batch.draft_cache_lengths,
            return_logits=ReturnLogits.LAST_TOKEN,
            return_hidden_states=self.step_hidden_mode,
        )
        n_devs = len(self.mesh.devices)
        # (last_logits, hs_0..hs_{n-1}).
        return Proposed(
            logits=outputs[0].local_shards[0],
            hidden=list(outputs[1 : 1 + n_devs]),
        )
