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
"""Inkling's target and chained MTP draft, as sequential driver adapters."""

from __future__ import annotations

from collections.abc import Sequence

from max.dtype import DType
from max.graph import (
    BufferType,
    DeviceRef,
    Dim,
    DimLike,
    TensorType,
    TensorValue,
    ops,
)
from max.nn.kernels import merge_ragged_tensors
from max.nn.kv_cache import PagedCacheValues
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.nn.transformer.distributed_transformer import (
    distributed_logits_postprocess,
)
from max.nn.transformer.transformer import logits_postprocess
from max.pipelines.speculative.driver import (
    CarryDimNames,
    DecodeKVSwap,
    DraftCache,
    DraftStepInput,
    Proposed,
    ReuseSpec,
    SequentialBatch,
)
from max.pipelines.speculative.ragged_token_merger import _shape_to_scalar
from max.pipelines.speculative.spec_target import Verified

from ..inkling.inkling import Inkling
from ..inkling.model_config import InklingConfig
from ..inkling.state_cache import ConvSite
from .inkling_mtp import InklingMultiTokenPredictor

__all__ = [
    "DRAFT_CONV_POOLS",
    "DRAFT_PRIMARY_KV",
    "IMAGE_EMBEDDINGS",
    "IMAGE_INDICES",
    "POSITIONS",
    "TARGET_AUX_KV",
    "TARGET_PRIMARY_KV",
    "TARGET_STATE",
    "InklingMTPProposer",
    "InklingTarget",
    "split_kv_by_flavor",
]

POSITIONS = "positions"
IMAGE_EMBEDDINGS = "image_embeddings"
IMAGE_INDICES = "image_indices"
TARGET_STATE = "target_state"
"""The backbone's conv-state leaves, drawn from the recurrent cache group."""
DRAFT_CONV_POOLS = "draft_conv_pools"
TARGET_AUX_KV = "target_passthrough_kv"
"""Every target cache leaf but the primary one, which rides ``kv_collections``.

Inkling splits attention across a global and a local cache, and the driver
carries exactly one target leaf; the rest reach the target through
:attr:`SequentialBatch.extra` so the adapter can hand back the by-flavor
dict its decoder layers index into.
"""
TARGET_PRIMARY_KV = "target_primary_kv"
"""Which flavor ``kv_collections`` holds."""
DRAFT_PRIMARY_KV = "draft_primary_kv"
"""Which flavor ``draft_kv_collections`` holds."""

_TargetHidden = list[TensorValue]
"""Per-device normalized hidden states, one entry per rank."""


def split_kv_by_flavor(
    by_flavor: dict[str, list[PagedCacheValues]],
) -> tuple[str, list[PagedCacheValues], dict[str, list[PagedCacheValues]]]:
    """Splits a by-flavor cache dict into the driver's primary leaf and the rest.

    The driver carries one cache leaf plus an aux mapping, while Inkling's
    decoder layers index a leaf per attention flavor. Returns the primary's
    name alongside it so the adapters rejoin the dict by name rather than by
    reproducing which flavor came first.
    """
    keys = list(by_flavor)
    return (
        keys[0],
        by_flavor[keys[0]],
        {key: by_flavor[key] for key in keys[1:]},
    )


def draft_row_inputs(
    row_offsets: Sequence[TensorValue],
) -> list[list[TensorValue]]:
    """Rows the draft convolves in: every batch item reads the single zero
    slot of :class:`InklingConvScratchPools`."""
    rows: list[list[TensorValue]] = []
    for offsets in row_offsets:
        row = ops.broadcast_to(
            ops.constant(0, DType.uint32, device=offsets.device),
            [offsets.shape[0] - 1],
        )
        rows.append([row] * len(ConvSite))
    return rows


def merge_positions(
    positions: TensorValue,
    input_row_offsets: TensorValue,
    draft_tokens: TensorValue,
) -> TensorValue:
    """Extends each sequence's position ramp across the appended draft tokens."""
    device = positions.device
    k = _shape_to_scalar(draft_tokens.shape[1], device, dtype=DType.uint32)
    last_pos = ops.gather(positions, input_row_offsets[1:] - 1, axis=0)
    batch_plus_1 = ops.shape_to_tensor([input_row_offsets.shape[0]])[0]
    indices = ops.range(
        start=0,
        stop=batch_plus_1,
        out_dim=input_row_offsets.shape[0],
        device=device,
        dtype=DType.uint32,
    )
    draft_offsets = indices * k
    k_range = ops.range(
        start=0,
        stop=ops.shape_to_tensor([draft_tokens.shape[1]])[0],
        out_dim=draft_tokens.shape[1],
        device=device,
        dtype=DType.uint32,
    )
    draft_pos = ops.reshape(
        ops.unsqueeze(last_pos, 1) + 1 + ops.unsqueeze(k_range, 0),
        [-1],
    )
    merged, _ = merge_ragged_tensors(
        positions, input_row_offsets, draft_pos, draft_offsets
    )
    return ops.rebind(merged, ["merged_seq_len"])


def _last_accepted_idx(
    merged_offsets: TensorValue,
    num_accepted: TensorValue,
    num_draft_tokens: Dim,
    device: DeviceRef,
) -> TensorValue:
    last_idx = merged_offsets[1:] - 1
    k = _shape_to_scalar(num_draft_tokens, device)
    return (
        ops.rebind(last_idx, ["batch_size"])
        - k.broadcast_to(["batch_size"])
        + num_accepted.cast(last_idx.dtype)
    )


class InklingTarget:
    """Inkling's verify pass, and the merged position ramp both phases read."""

    def __init__(self, target: Inkling, config: InklingConfig) -> None:
        self.target = target
        self.config = config
        self._merged_positions: TensorValue | None = None

    def merged_positions(self, batch: SequentialBatch) -> TensorValue:
        """The position ramp extended across each request's draft tokens.

        Both the target and the draft place their rows against this same ramp,
        and it is built from inputs the batch already carries, so either phase
        can be the one that asks for it first.
        """
        if self._merged_positions is None:
            self._merged_positions = merge_positions(
                batch.extra[POSITIONS],
                batch.input_row_offsets,
                batch.draft_tokens,
            )
        return self._merged_positions

    def kv_by_flavor(
        self, batch: SequentialBatch
    ) -> dict[str, list[PagedCacheValues]]:
        """The target's caches keyed the way its decoder layers index them."""
        return {
            batch.extra[TARGET_PRIMARY_KV]: batch.kv_collections,
            **batch.extra[TARGET_AUX_KV],
        }

    def verify(self, batch: SequentialBatch) -> Verified[_TargetHidden]:
        outputs = self.target(
            batch.merged_tokens,
            batch.merged_offsets,
            self.merged_positions(batch),
            batch.return_n_logits,
            batch.extra[IMAGE_EMBEDDINGS],
            batch.extra[IMAGE_INDICES],
            batch.signal_buffers,
            self.kv_by_flavor(batch),
            batch.extra[TARGET_STATE],
        )
        # VARIABLE logits + ALL_NORMALIZED hidden states ->
        # (last_logits, logits, offsets, hidden per device...).
        return Verified(
            logits=outputs[1], hidden=list(outputs[3 : 3 + batch.n_devs])
        )

    def ep_input_types(self) -> Sequence[TensorType | BufferType]:
        return ()


class InklingMTPProposer:
    """The chained MTP depths: one full decoder block per speculative step."""

    reuse: ReuseSpec | None = None
    decode_swaps: tuple[DecodeKVSwap, ...] = ()
    draft_cache = DraftCache.PER_DEPTH
    split_prefix = "mtp"
    carry_dim_names = CarryDimNames(prefix="mtp_step")
    # forward_depth returns one tensor per rank, never allgathered.
    step_hidden_mode = ReturnHiddenStates.ALL
    uses_thinking_phase = True

    def __init__(
        self, draft: InklingMultiTokenPredictor, target: InklingTarget
    ) -> None:
        self.draft = draft
        self.target = target
        self.hidden_dim: DimLike = target.config.text_config.hidden_size
        self.passthrough_decode_swaps: tuple[str, ...] = tuple(
            list(draft.kv_params.children)[1:]
        )
        self._decode_base: TensorValue | None = None

    def _kv_by_flavor(
        self, batch: SequentialBatch
    ) -> dict[str, list[PagedCacheValues]]:
        return {
            batch.extra[DRAFT_PRIMARY_KV]: batch.draft_kv_collections,
            **batch.passthrough_kv,
        }

    def _decode_positions(
        self, batch: SequentialBatch, index: int
    ) -> TensorValue:
        """Where step ``index`` sits: the accepted token, plus one per step."""
        if self._decode_base is None:
            assert batch.num_accepted is not None
            self._decode_base = ops.gather(
                self.target.merged_positions(batch),
                _last_accepted_idx(
                    batch.merged_offsets,
                    batch.num_accepted,
                    batch.num_draft_tokens,
                    batch.device0,
                ),
                axis=0,
            )
        return self._decode_base + index

    def _depth_logits(
        self, batch: SequentialBatch, hidden: _TargetHidden, *, variable: bool
    ) -> TensorValue:
        """LM-head over a depth's hidden states, through the target's head."""
        config = self.target.config
        target = self.target.target
        return_logits = (
            ReturnLogits.VARIABLE if variable else ReturnLogits.LAST_TOKEN
        )
        logits_scaling = config.text_config.logits_mup_width_multiplier
        if batch.n_devs > 1:
            outputs = distributed_logits_postprocess(
                hidden,
                batch.query_offsets_per_dev,
                batch.return_n_logits,
                lm_head=target._distributed_lm_head,
                signal_buffers=batch.signal_buffers,
                return_logits=return_logits,
                device=batch.device0,
                norm_shards=target.norm_shards,
                logits_scaling=logits_scaling,
            )
        else:
            outputs = logits_postprocess(
                hidden[0],
                batch.query_offsets_per_dev[0],
                batch.return_n_logits,
                target.norm,
                target._lm_head,
                return_logits,
                logits_scaling=logits_scaling,
            )
        # VARIABLE emits (last_token_logits, logits, offsets); LAST_TOKEN emits
        # the single logits tensor first.
        return outputs[1] if variable else outputs[0]

    def prefill(
        self,
        batch: SequentialBatch,
        tokens: TensorValue,
        target_hidden: _TargetHidden,
    ) -> Proposed:
        draft_rows = draft_row_inputs(batch.query_offsets_per_dev)
        hidden = self.draft.forward_depth(
            0,
            self.draft.embed_tokens(tokens, batch.signal_buffers),
            target_hidden,
            self._kv_by_flavor(batch),
            batch.query_offsets_per_dev,
            self.target.merged_positions(batch),
            batch.extra[DRAFT_CONV_POOLS],
            draft_rows,
            batch.signal_buffers,
        )
        return Proposed(
            logits=self._depth_logits(batch, hidden, variable=True),
            hidden=hidden,
        )

    def step(
        self, batch: SequentialBatch, draft_input: DraftStepInput, index: int
    ) -> Proposed:
        step_dim = f"{self.carry_dim_names.prefix}{index}_batch"
        draft_rows = draft_row_inputs(batch.query_offsets_per_dev)
        embeds = [
            embed.rebind([step_dim, self.hidden_dim])
            for embed in self.draft.embed_tokens(
                draft_input.tokens, batch.signal_buffers
            )
        ]
        hidden = self.draft.forward_depth(
            index,
            embeds,
            draft_input.hidden,
            self._kv_by_flavor(batch),
            batch.query_offsets_per_dev,
            self._decode_positions(batch, index).rebind([step_dim]),
            batch.extra[DRAFT_CONV_POOLS],
            draft_rows,
            batch.signal_buffers,
        )
        return Proposed(
            logits=self._depth_logits(batch, hidden, variable=False),
            hidden=hidden,
        )
