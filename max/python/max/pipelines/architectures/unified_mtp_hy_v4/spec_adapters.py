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
"""Speculative-decoding adapters between the shared driver and Hy4.

The driver speaks graph values and graph-API caches; the Hy4 target and draft
are ModuleV3. Each adapter wraps its inputs as ``Tensor``s, runs the module,
and hands graph values back.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from max.experimental import functional as F
from max.experimental.nn import Module as ModuleV3
from max.experimental.nn.common_layers.kv_cache import PagedCacheValues
from max.experimental.sharding import DeviceMapping, DeviceMesh, Replicated
from max.experimental.tensor import Tensor
from max.graph import BufferType, DimLike, TensorType, TensorValue
from max.nn.kv_cache import PagedCacheValues as GraphPagedCacheValues
from max.nn.layer import Module
from max.nn.transformer import ReturnHiddenStates
from max.pipelines.speculative.driver import (
    CarryDimNames,
    DecodeKVSwap,
    DraftCache,
    DraftStepInput,
    Proposed,
    ReuseSpec,
    SequentialBatch,
    gather_variable_rows,
)
from max.pipelines.speculative.spec_target import Verified

from ..hy_v4.hy_v4 import HYV4TextModel
from .hy_v4_mtp import HYV4MultiTokenPredictor

__all__ = [
    "INDEXER_DRAFT_KV",
    "INDEXER_TARGET_KV",
    "HYV4MTPProposer",
    "HYV4Target",
    "ModuleV3Handle",
]

INDEXER_TARGET_KV = "target_indexer"
"""Names the target's indexer K cache leaf in ``passthrough_kv``."""

INDEXER_DRAFT_KV = "draft_indexer"
"""Names the draft's indexer K cache leaf in ``passthrough_kv``."""


class ModuleV3Handle(Module):
    """Lets the driver register a ModuleV3 module it never calls.

    The driver keeps its target and draft as graph-API sublayers for their
    state dicts; ModuleV3 names and loads these weights itself.
    """

    def __init__(self, module: ModuleV3[..., Any]) -> None:
        super().__init__()
        self.module = module

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.module(*args, **kwargs)


def _caches(
    mla: Sequence[GraphPagedCacheValues],
    indexer: Sequence[GraphPagedCacheValues],
    mapping: DeviceMapping,
) -> tuple[PagedCacheValues, PagedCacheValues]:
    return (
        PagedCacheValues.from_upstream(mla, mapping),
        PagedCacheValues.from_upstream(indexer, mapping),
    )


def _replicated(values: Sequence[TensorValue], mesh: DeviceMesh) -> Tensor:
    """Wraps the driver's per-device copies of one tensor."""
    return Tensor.from_shard_values(
        values, DeviceMapping(mesh, (Replicated(),))
    )


def _on_mesh(value: TensorValue, mesh: DeviceMesh) -> Tensor:
    """Copies a device-0 value to every device, as ``HYV4.forward`` does."""
    t = Tensor.from_graph_value(value)
    return F.distributed_broadcast(t, mesh) if mesh.num_devices > 1 else t


def _per_device(t: Tensor) -> list[TensorValue]:
    return [TensorValue(shard) for shard in t.local_shards]


def _first_device(t: Tensor) -> TensorValue:
    """Device 0's copy of a replicated tensor, where the driver samples."""
    return _per_device(t)[0]


def _batch_context_length(batch: SequentialBatch) -> Tensor:
    """The host prefill buffer length the attention blocks take instead of
    reading the plan's back. It counts no draft rows; the sparse MLA kernels,
    the only ones Hy4 builds, never read it."""
    (length,) = batch.batch_context_lengths
    return Tensor.from_graph_value(length)


class HYV4Target:
    """Runs the Hy4 target over the merged window.

    Returns the logits of the last ``return_n_logits`` positions per request,
    on the first device, and each device's copy of the final normalized
    hidden state of every position.
    """

    def __init__(self, target: HYV4TextModel) -> None:
        self.target = target

    def verify(self, batch: SequentialBatch) -> Verified[list[TensorValue]]:
        mesh = self.target.embed_tokens.weight.mesh
        tokens = _on_mesh(batch.merged_tokens, mesh)
        offsets = _replicated(batch.merged_offsets_per_dev, mesh)
        mla, indexer = _caches(
            batch.kv_collections,
            batch.passthrough_kv[INDEXER_TARGET_KV],
            tokens.mapping,
        )
        hidden = self.target.head(
            self.target.streams(
                tokens, mla, indexer, offsets, _batch_context_length(batch)
            )
        )
        n = Tensor.from_graph_value(batch.return_n_logits)
        rows, _ = gather_variable_rows(hidden, offsets, n, mesh)
        logits = self.target.logits(rows)
        return Verified(
            logits=_first_device(logits), hidden=_per_device(hidden)
        )

    def ep_input_types(self) -> Sequence[TensorType | BufferType]:
        return ()


class HYV4MTPProposer:
    """The single Hy4 MTP layer, run once per draft step.

    Step 0 runs the draft's own indexer over the corrected window; steps
    1..K-1 reuse its top-k at each request's accepted position.
    """

    decode_swaps: tuple[DecodeKVSwap, ...] = (
        DecodeKVSwap.MAX_PROMPT_LENGTH_ONE,
        DecodeKVSwap.DRAFT_ATTENTION_DISPATCH_METADATA,
        DecodeKVSwap.DRAFT_MLA_NUM_PARTITIONS,
    )
    passthrough_decode_swaps: tuple[str, ...] = ()
    draft_cache = DraftCache.OWN
    split_prefix = "mtp"
    carry_dim_names = CarryDimNames()
    step_hidden_mode = ReturnHiddenStates.ALL
    uses_thinking_phase = False
    supports_zero_draft_rows = False

    def __init__(
        self, draft: HYV4MultiTokenPredictor, target: HYV4TextModel
    ) -> None:
        self.draft = draft
        self.target = target
        self.hidden_dim: DimLike = int(draft.enorm.dim)
        self.reuse: ReuseSpec | None = ReuseSpec(
            dim=draft.self_attn.index_topk, split_prefix="mtp_topk"
        )

    def _predict(
        self,
        batch: SequentialBatch,
        tokens: TensorValue,
        hidden: Sequence[TensorValue],
        prev_topk_indices: Sequence[TensorValue] | None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Returns the draft's hidden state, its top-k and its row offsets.

        ``hidden`` and ``prev_topk_indices`` hold one copy per device.
        """
        mesh = self.target.embed_tokens.weight.mesh
        token_ids = _on_mesh(tokens, mesh)
        offsets = _replicated(batch.query_offsets_per_dev, mesh)
        mla, indexer = _caches(
            batch.draft_kv_collections,
            batch.passthrough_kv[INDEXER_DRAFT_KV],
            token_ids.mapping,
        )
        out, topk = self.draft.predict(
            self.target.embed_tokens,
            token_ids,
            _replicated(hidden, mesh),
            mla,
            indexer,
            self.target.rope.freqs_cis.to(mesh),
            offsets,
            None
            if prev_topk_indices is None
            else _replicated(prev_topk_indices, mesh),
            _batch_context_length(batch),
        )
        return out, topk, offsets

    def prefill(
        self,
        batch: SequentialBatch,
        tokens: TensorValue,
        target_hidden: list[TensorValue],
    ) -> Proposed:
        out, topk, offsets = self._predict(batch, tokens, target_hidden, None)
        n = Tensor.from_graph_value(batch.return_n_logits)
        rows, _ = gather_variable_rows(out, offsets, n, out.mesh)
        logits = self.target.logits(rows)
        return Proposed(
            logits=_first_device(logits),
            hidden=_per_device(out),
            reuse=_per_device(topk),
        )

    def step(
        self, batch: SequentialBatch, draft_input: DraftStepInput, index: int
    ) -> Proposed:
        del index
        out, _, _ = self._predict(
            batch, draft_input.tokens, draft_input.hidden, draft_input.reuse
        )
        return Proposed(
            logits=_first_device(self.target.logits(out)),
            hidden=_per_device(out),
        )
