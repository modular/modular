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
"""Inkling's call, and the chained MTP depths its checkpoint ships.

Inkling's own batch processor prepares each batch: it stages the tokens'
positions and encodes the batch's new images, as when Inkling serves alone.
Both ride past the canonical tail. The depths embed and project through the
target's embedding and head, and each depth runs one full decoder block.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from max.driver import Buffer
from max.dtype import DType
from max.graph import (
    BufferType,
    DeviceRef,
    Dim,
    DimLike,
    TensorType,
    TensorValue,
    Value,
    ops,
)
from max.nn.kernels import merge_ragged_tensors
from max.nn.kv_cache import PagedCacheValues
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.nn.transformer.distributed_transformer import (
    distributed_logits_postprocess,
)
from max.nn.transformer.transformer import logits_postprocess
from max.pipelines.speculative.block_driver import BlockBatch
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
from max.pipelines.speculative.spec_adapters.target_adapter import TargetAdapter
from max.pipelines.speculative.spec_target import Verified

if TYPE_CHECKING:
    from max.pipelines.speculative.config import SpeculativeConfig
    from max.pipelines.speculative.driver import SequentialProposer
    from max.pipelines.speculative.spec_adapters import VerifyFn

PRIMARY_KV = "full_attention"

_STATE_KV = "conv_state"


POSITIONS = "positions"
IMAGE_EMBEDDINGS = "image_embeddings"
IMAGE_INDICES = "image_indices"
"""Keys the inputs Inkling's batch processor prepares ride under in a driver
batch's ``extra``."""

_DRAFT = "draft_"
"""Prefixes the draft's leaves beside its primary one in ``passthrough_kv``."""


def trailing_input_types(target_config: Any) -> list[TensorType | BufferType]:
    """Returns the positions and image inputs Inkling's own graph declares.

    That is the positions, the image rows the tower encoded and the token
    rows they replace.
    """
    device = target_config.devices[0]
    return [
        TensorType(DType.uint32, shape=["total_seq_len"], device=device),
        TensorType(
            target_config.dtype,
            shape=["total_image_tokens", target_config.text_config.hidden_size],
            device=device,
        ),
        TensorType(DType.int32, shape=["total_image_tokens"], device=device),
    ]


def extra(target_config: Any, trailing: Sequence[Value[Any]]) -> dict[str, Any]:
    """Names the inputs :func:`trailing_input_types` declares."""
    del target_config
    positions, embeddings, indices = (value.tensor for value in trailing[:3])
    return {
        POSITIONS: positions,
        IMAGE_EMBEDDINGS: embeddings,
        IMAGE_INDICES: indices,
    }


def batch_trailing_buffers(target_inputs: Any) -> list[Buffer]:
    """Returns the :func:`trailing_input_types` inputs Inkling prepared."""
    return [
        target_inputs.positions,
        target_inputs.image_embeddings,
        target_inputs.image_indices,
    ]


def verify(
    target: Any,
    target_config: Any,
    hidden_states: ReturnHiddenStates,
    draft_config: Any,
) -> VerifyFn:
    """Returns the verify of an Inkling model, on every device it spans.

    Raises:
        ValueError: If the target wasn't built to return its last
            normalized layer, which the MTP depths read.
    """
    del draft_config
    if hidden_states != ReturnHiddenStates.ALL_NORMALIZED:
        raise ValueError(
            "An Inkling target verifies only for drafts that read its last"
            f" normalized layer. Got {hidden_states.value}."
        )
    # A layer subgraph takes its weights by a prefix from the module's root,
    # and the fused graph nests the module under ``target.``.
    target.use_subgraphs = False
    return _Verify(target, target_config)


def sequential_proposer(
    verify: VerifyFn,
    draft: Any,
    draft_config: Any,
    speculative: SpeculativeConfig,
) -> SequentialProposer[list[TensorValue]]:
    """Returns the proposer of the MTP depths, on the verify's target."""
    del draft_config, speculative
    assert isinstance(verify, _Verify)
    return _MTPProposer(draft, verify)


def _merge_positions(
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


class _Verify:
    """Inkling's verify pass, and the merged position ramp both phases read."""

    def __init__(self, target: Any, config: Any) -> None:
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
            self._merged_positions = _merge_positions(
                batch.extra[POSITIONS],
                batch.input_row_offsets,
                batch.draft_tokens,
            )
        return self._merged_positions

    def __call__(
        self, batch: SequentialBatch | BlockBatch
    ) -> Verified[list[TensorValue]]:
        assert isinstance(batch, SequentialBatch)
        # The decoder layers index the target's caches by flavor, and read
        # the conv state's leaves per device.
        outputs = self.target(
            batch.merged_tokens,
            batch.merged_offsets,
            self.merged_positions(batch),
            batch.return_n_logits,
            batch.extra[IMAGE_EMBEDDINGS],
            batch.extra[IMAGE_INDICES],
            batch.signal_buffers,
            {PRIMARY_KV: batch.kv_collections, **batch.passthrough_kv},
            batch.passthrough_kv[_STATE_KV],
        )
        return Verified(
            logits=outputs.logits["logits"],
            hidden=list(
                outputs.hidden_states[ReturnHiddenStates.ALL_NORMALIZED.value]
            ),
        )


class _MTPProposer:
    """The chained MTP depths: one full decoder block per speculative step."""

    reuse: ReuseSpec | None = None
    decode_swaps: tuple[DecodeKVSwap, ...] = ()
    # TODO(thomas.borstad): give the draft its own dispatch metadata instead
    # of replaying the verify metadata over zeroed cache lengths; that
    # changes the graph's input signature.
    draft_cache = DraftCache.PER_DEPTH
    split_prefix = "mtp"
    carry_dim_names = CarryDimNames(prefix="mtp_step")
    # forward_depth returns one tensor per rank, never allgathered.
    step_hidden_mode = ReturnHiddenStates.ALL
    uses_thinking_phase = True
    supports_zero_draft_rows = False

    def __init__(self, draft: Any, verify: _Verify) -> None:
        self.draft = draft
        self.verify = verify
        target = verify.target
        draft.embed = target.embed
        draft.backbone_embed_norm_shards = target.embed_norm_shards
        self.hidden_dim: DimLike = verify.config.text_config.hidden_size
        flavors = list(draft.kv_params.children)
        # The unified model drives the draft's leaf of the target's primary
        # flavor, or its first when the depths have none of that flavor.
        self._primary = PRIMARY_KV if PRIMARY_KV in flavors else flavors[0]
        self.passthrough_decode_swaps: tuple[str, ...] = tuple(
            f"{_DRAFT}{flavor}" for flavor in flavors if flavor != self._primary
        )
        self._decode_base: TensorValue | None = None

    def _kv_by_flavor(
        self, batch: SequentialBatch
    ) -> dict[str, list[PagedCacheValues]]:
        return {
            self._primary: batch.draft_kv_collections,
            **{
                name.removeprefix(_DRAFT): leaves
                for name, leaves in batch.passthrough_kv.items()
                if name.startswith(_DRAFT)
            },
        }

    def _decode_positions(
        self, batch: SequentialBatch, index: int
    ) -> TensorValue:
        """Where step ``index`` sits: the accepted token, plus one per step."""
        if self._decode_base is None:
            assert batch.num_accepted is not None
            self._decode_base = ops.gather(
                self.verify.merged_positions(batch),
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
        self,
        batch: SequentialBatch,
        hidden: list[TensorValue],
        *,
        variable: bool,
    ) -> TensorValue:
        """LM-head over a depth's hidden states, through the target's head."""
        target = self.verify.target
        return_logits = (
            ReturnLogits.VARIABLE if variable else ReturnLogits.LAST_TOKEN
        )
        logits_scaling = (
            self.verify.config.text_config.logits_mup_width_multiplier
        )
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
        return outputs.logits["logits" if variable else "next_token_logits"]

    def prefill(
        self,
        batch: SequentialBatch,
        tokens: TensorValue,
        target_hidden: list[TensorValue],
    ) -> Proposed:
        hidden = self.draft.forward_depth(
            0,
            self.draft.embed_tokens(tokens, batch.signal_buffers),
            target_hidden,
            self._kv_by_flavor(batch),
            batch.query_offsets_per_dev,
            self.verify.merged_positions(batch),
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
            batch.signal_buffers,
        )
        return Proposed(
            logits=self._depth_logits(batch, hidden, variable=False),
            hidden=hidden,
        )


ADAPTER = TargetAdapter(
    primary_kv=PRIMARY_KV,
    verify=verify,
    trailing_input_types=trailing_input_types,
    batch_trailing_buffers=batch_trailing_buffers,
    extra=extra,
    sequential_proposer=sequential_proposer,
)
