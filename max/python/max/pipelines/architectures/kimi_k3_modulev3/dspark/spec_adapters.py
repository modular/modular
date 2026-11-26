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
"""Kimi K3's call beside the DSpark drafter (ModuleV3).

Verifying a speculative window advances every KDA recurrence and slides no
length pointer back, so the verify runs against the ring
:mod:`..spec_state` binds on the KDA layers: it records each token's
recurrent update there and leaves the conv window unwritten, then the
accepted records are folded into the live pools once the accepted count is
known. A window with no drafts lands both live pools itself, so one graph
serves the prefill step too.

This is the target's call, but it lives in the drafter's package: the
adapter needs the drafter's proposer beside the target's rollback, and the
drafter's package already depends on the target's, while a target package
cannot depend on its draft's.

The target is a Tensor-API ModuleV3 module while the block driver runs on
graph values, so the verify converts at its own boundary:
``Tensor.from_graph_value`` on the way in, one fused tap tensor per device
on the way out -- the drafter reads those through its ``context_proj``.

Both calls size their MLA prefill plans from the batch's host context
lengths, which the verify's distributed inputs bring; :mod:`.mla_extent`
selects the capture-safe value each call reads, so neither plan reads a
device extent back and the fused graph stays capturable.

The KDA state rides the driver batch's ``passthrough_kv`` under the cache
tree's ``state`` child, per device: the conv and recurrent leaves the model
mutates, then the verify ring. The model's ``forward`` unflattens the same
graph inputs itself, so :func:`target_kv_inputs` rebuilds the flat list the
forward re-consumes; the pools it hands the fold are the very buffers the
forward mutates, which ``test_spec_adapter_k3.py`` pins.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from functools import partial
from typing import TYPE_CHECKING, Any

from max import tree
from max.dtype import DType
from max.graph import (
    BufferType,
    BufferValue,
    TensorType,
    TensorValue,
)
from max.nn.kv_cache import (
    KVCacheInputsPerDevice,
    MLAKVCacheParams,
    MultiKVCacheParams,
    RecurrentStateInputsPerDevice,
    RecurrentStateParams,
)
from max.nn.legacy.comm import Signals
from max.nn.outputs import ReturnHiddenStates
from max.pipelines.speculative.block_driver import BlockBatch
from max.pipelines.speculative.spec_adapters.target_adapter import TargetAdapter
from max.pipelines.speculative.spec_target import Verified
from max.tensor import Tensor

from ..kimi_k3 import KimiK3Decoder
from ..model import KimiK3Model
from ..spec_state import KimiK3RecurrentState
from ..state_cache import (
    ATTN_CACHE_KEY,
    CONV_LEAF_ID,
    RECURRENT_LEAF_ID,
    STATE_CACHE_KEY,
    VERIFY_RING_LEAF_ID,
    attn_cache,
    has_verify_ring,
)
from .mla_extent import k3_spec_mla_extent
from .proposer import K3DSparkProposer, k3_dspark_block_proposer

if TYPE_CHECKING:
    from max.pipelines.speculative.config import SpeculativeConfig
    from max.pipelines.speculative.driver import SequentialBatch
    from max.pipelines.speculative.spec_adapters import VerifyFn

__all__ = [
    "ADAPTER",
    "KimiK3BlockTarget",
    "target_kv_inputs",
]

_StateInputs = Sequence[RecurrentStateInputsPerDevice[TensorValue, BufferValue]]
"""Per-device KDA state leaves: conv, recurrent, then the verify ring."""


def _state_leaves(batch: BlockBatch) -> _StateInputs:
    """Returns the batch's per-device KDA state leaves, ring asserted.

    Raises:
        ValueError: If the cache declares no verify ring -- a non-speculative
            tree, whose live leaves the verify would advance past the anchor
            with no way back.
    """
    state: list[RecurrentStateInputsPerDevice[TensorValue, BufferValue]] = []
    for device_state in batch.passthrough_kv[STATE_CACHE_KEY]:
        assert isinstance(device_state, RecurrentStateInputsPerDevice)
        if not has_verify_ring(device_state):
            raise ValueError(
                "A speculative Kimi K3 verify needs the cache's verify ring:"
                f" the state child declares {len(device_state.leaves)} leaves."
            )
        state.append(device_state)
    if len(state) != batch.n_devs:
        raise ValueError(
            f"The KDA state must reach every one of {batch.n_devs} devices;"
            f" got {len(state)}."
        )
    return state


def _require_own_draft_leaves(batch: BlockBatch) -> None:
    """Refuses a drafter that would write its context into the target's KV.

    The block driver hands the draft its own leaf because the drafter's
    ``construct_kv_params`` declares one; a registry or pairing change that
    dropped it would leave both models writing the same pages.
    """
    for target_kv, draft_kv in zip(
        batch.kv_collections, batch.draft_kv_collections, strict=True
    ):
        assert isinstance(target_kv, KVCacheInputsPerDevice)
        assert isinstance(draft_kv, KVCacheInputsPerDevice)
        if target_kv.kv_blocks is draft_kv.kv_blocks:
            raise ValueError(
                "The DSpark drafter needs its own KV leaf: the draft and the"
                " target share one pool."
            )


def target_kv_inputs(target: KimiK3Model, batch: BlockBatch) -> list[Tensor]:
    """Returns the target's cache tree, flattened for its ``forward``.

    The forward unflattens ``*kv_inputs`` through its own
    ``kv_params.unflatten_kv_inputs``, consuming the values in child
    declaration order; this rebuilds that list out of the per-device leaves
    the driver hands the batch, which the graph's decode built from the same
    input values. Wrapping never copies: each returned tensor's graph value
    is the leaf it wraps.
    """
    per_child: dict[str, Sequence[Any]] = {
        ATTN_CACHE_KEY: batch.kv_collections,
        STATE_CACHE_KEY: batch.passthrough_kv[STATE_CACHE_KEY],
    }
    assert isinstance(target.kv_params, MultiKVCacheParams), (
        "a speculatively served Kimi K3 holds a cache tree with a state child"
    )
    children = target.kv_params.children
    leaves, _ = tree.flatten(
        OrderedDict((name, tuple(per_child[name])) for name in children)
    )
    return [Tensor.from_graph_value(leaf) for leaf in leaves]


class KimiK3BlockTarget:
    """Verifies the merged window, recording the KDA state for rollback.

    Carries the built target module so the drafter's proposer can score its
    block through the target's ``lm_head_weight``, beside the
    :class:`~..spec_state.KimiK3RecurrentState` that binds the verify's
    ring.
    """

    # Brings the batch's host context lengths, which size the MLA plans.
    takes_distributed_inputs = True

    def __init__(self, target: KimiK3Model) -> None:
        self.target = target
        self.state = KimiK3RecurrentState.from_model(target)
        # A verify runs against the pairing's cache, but the same module
        # builds without one (the padded and tap-test forms), where no MLA
        # plan exists to size. A decompressed cache builds no plan either,
        # so its verify reads no extent.
        params = (
            attn_cache(target.kv_params)
            if target.kv_params is not None
            else None
        )
        self._mla_params = (
            params if isinstance(params, MLAKVCacheParams) else None
        )

    def __call__(
        self, batch: SequentialBatch | BlockBatch
    ) -> Verified[list[TensorValue]]:
        """Runs the target over the merged window, under the ring capture.

        Raises:
            ValueError: If the cache declares no verify ring, or the draft
                shares the target's KV leaf.
        """
        assert isinstance(batch, BlockBatch)
        _require_own_draft_leaves(batch)
        state = _state_leaves(batch)
        extent = self._mla_extent(batch) if self._mla_params else None
        with self.state.capturing(batch.num_draft_tokens, state):
            outputs = self.target(
                Tensor.from_graph_value(batch.merged_tokens),
                Tensor.from_graph_value(batch.return_n_logits),
                Tensor.from_graph_value(batch.merged_offsets),
                *target_kv_inputs(self.target, batch),
                *(() if extent is None else (Tensor.from_graph_value(extent),)),
            )
        assert outputs.logits is not None, (
            "the pairing asks the target for VARIABLE logits"
        )
        assert outputs.hidden_states is not None, (
            "the pairing asks the target for the tapped layers' states"
        )
        return Verified(
            logits=TensorValue(outputs.logits),
            hidden=_taps_per_device(outputs.hidden_states),
            roll_forward=partial(_roll_forward, self.state),
        )

    def _mla_extent(self, batch: BlockBatch) -> TensorValue:
        """The selected host extent the verify's MLA prefill plan reads.

        A prompt step's extent is the batch's context length plus an
        allowance per request; the select keeps the captured graphs on the
        batch-size-keyed bound instead, so no plan reads a device-computed
        extent back.
        """
        params = self._mla_params
        assert params is not None, "an extent needs the target's MLA cache"
        return k3_spec_mla_extent(
            context_length=batch.batch_context_lengths[0],
            reach=0,
            kv=batch.kv_collections[0],
            input_row_offsets=batch.input_row_offsets,
            num_draft_tokens=params.num_draft_tokens,
            page_size=params.page_size,
            mla_buffer_size=self.target.mla_buffer_size,
        )


def _roll_forward(state: KimiK3RecurrentState, batch: BlockBatch) -> None:
    """Lands the KDA state on the accepted prefix.

    The driver calls this with its batch once the accepted count is set,
    before the drafter reads the live pools; the state leaves it reads are
    the very buffers the forward just left pre-verify.
    """
    assert batch.num_accepted is not None
    state.roll_forward(
        _state_leaves(batch),
        merged_offsets=batch.merged_offsets,
        merged_offsets_per_dev=batch.merged_offsets_per_dev,
        num_accepted=batch.num_accepted,
        signal_buffers=batch.signal_buffers,
        device=batch.device0,
    )


def _taps_per_device(taps: Tensor) -> list[TensorValue]:
    """Returns the fused tap tensor as one graph value per device.

    The taps stay one replicated tensor across the mesh (the drafter's
    ``context_proj`` runs on every device), so a TP build's copy becomes one
    entry per device and a single-device build stays one entry.
    """
    if taps.is_distributed:
        return [TensorValue(shard) for shard in taps.local_shards]
    return [TensorValue(taps)]


def verify(
    target: Any,
    target_config: Any,
    hidden_states: ReturnHiddenStates,
    draft_config: Any,
) -> VerifyFn:
    """Returns the verify of a Kimi K3 target, on every device it runs on.

    Accepts the constructed decoder the registered builder hands the
    pairing, or the bare tower the parity harness builds itself.

    Raises:
        ValueError: If the target wasn't built to return the layers its
            draft names, or holds no KDA layer to record.
    """
    del target_config, draft_config
    if hidden_states != ReturnHiddenStates.SELECTED_LAYERS:
        raise ValueError(
            "A Kimi K3 target verifies only for drafts that read its"
            f" selected layers. Got {hidden_states.value}."
        )
    if isinstance(target, KimiK3Decoder):
        # The registered builder hands the pairing the constructed decoder;
        # the tower its model holds is what the verify drives.
        target = target.model
    assert isinstance(target, KimiK3Model)
    return KimiK3BlockTarget(target)


def block_proposer(
    verify: VerifyFn,
    draft: Any,
    draft_config: Any,
    speculative: SpeculativeConfig,
) -> K3DSparkProposer:
    """Returns the DSpark proposer of the built drafter, on this verify.

    The drafter holds no head of its own, so its proposer reads the built
    target this verify carries -- the module whose ``lm_head_weight`` the
    block scores through.
    """
    assert isinstance(verify, KimiK3BlockTarget)
    return k3_dspark_block_proposer(verify, draft, draft_config, speculative)


def export_contract(
    speculative: SpeculativeConfig,
    draft_config: Any,
    target_config: Any,
    kv_params: MultiKVCacheParams,
    input_types: Sequence[TensorType | BufferType],
    structured_output: bool,
) -> dict[str, object] | None:
    """Records the frozen export contract's resolved geometry.

    The record carries what Mach's engine checks the export's inputs
    against, independently of the graph
    (mach/docs/models/kimi-k3-inferact-dspark.md).

    Called from ``_driver_call`` on every fused build of this pairing,
    serving included -- so a build MAX legitimately serves must never raise
    here. A pairing outside the frozen contract (a decompressed, non-MLA
    target cache, a target with no KDA state, or no verify ring) records
    nothing: MAX keeps serving, and Mach's exporter fails its own
    "records no export contract" refusal rather than shipping a half-frozen
    record. The traced signal buffers the compile appends past the declared
    inputs are recorded by the exporter, which splices their types back in.

    Raises:
        ValueError: Only on a state MAX cannot build: a drafter whose block
            is not its draft width, which ``_driver_call`` refuses earlier.
    """
    del target_config
    devices = [device.id for device in kv_params.devices]
    tp = len(devices)
    target_tree = kv_params.children.get("target")
    if not isinstance(target_tree, MultiKVCacheParams):
        # No KDA state child: a target whose layers are all full attention.
        return None
    state_child = target_tree.children.get(STATE_CACHE_KEY)
    if not isinstance(state_child, RecurrentStateParams):
        return None
    regions = {region.leaf_id: region for region in state_child.regions}
    conv, recurrent = regions[CONV_LEAF_ID], regions[RECURRENT_LEAF_ID]
    ring = regions.get(VERIFY_RING_LEAF_ID)
    target_attn = attn_cache(target_tree)
    draft_attn = kv_params.children.get("draft")
    if (
        not isinstance(target_attn, MLAKVCacheParams)
        or not isinstance(draft_attn, MLAKVCacheParams)
        or ring is None
        or not ring.scratch
    ):
        # Outside the frozen contract: the decompressed target cache a
        # narrow-latent build serves through (the fused kernels' fallback,
        # not the frozen 576-wide latent), or a cache that declares no
        # verify-ring scratch. The Rust validator would refuse the record,
        # so none is written.
        return None

    def cache(params: MLAKVCacheParams) -> dict[str, object]:
        return {
            "kind": "mla",
            "layers": params.num_layers,
            "kv_heads": 1,
            "head_dim": params.head_dim,
            "page_size": params.page_size,
            "dtype": str(params.dtype),
        }

    width = speculative.draft_width
    block = draft_config.block_size
    anchor = K3DSparkProposer.samples_from_anchor
    if block != width or not anchor:
        raise ValueError(
            "The Inferact K3 drafter drafts its whole block, anchor included:"
            f" block_size {block} must equal the draft width {width}"
        )
    draft_proposal = speculative.draft_proposal
    return {
        "devices": devices,
        # The trace appends one signal buffer per device past every declared
        # input; the class constant sizes it (Signals.NUM_BYTES).
        "signal_bytes": Signals.NUM_BYTES,
        "target": cache(target_attn),
        "draft": cache(draft_attn),
        "state": {
            "layers": conv.num_layers,
            "conv_dim": conv.row_shape[0] * tp,
            "conv_span": conv.row_shape[1],
            "heads": recurrent.row_shape[0] * tp,
            "head_dim": recurrent.row_shape[1],
            "conv_dtype": str(conv.dtype),
            "recurrent_dtype": str(recurrent.dtype),
            "ring": {
                "length": ring.row_shape[1],
                "record_stride": ring.row_shape[2],
                "dtype": str(ring.dtype),
                "scratch": True,
            },
        },
        "spec_k": width,
        "block_size": block,
        "verify_width": width + 1,
        "sample_from_anchor": anchor,
        # The acceptance the built driver verifies with, which the sidecar's
        # own top-level record must agree with: greedy rows on a greedy
        # export, per-row sampled rows on a stochastic one.
        "acceptance_mode": (
            "greedy" if speculative.use_greedy_acceptance else "stochastic"
        ),
        # How the drafter proposes. ``sampled`` declares the draft
        # distributions beside the drafts -- one ``[batch, K, vocab]``
        # float32 input on device 0 -- and returns its own as the graph's
        # fourth output, since a step verifies drafts the step before it
        # drew: they cannot be produced and consumed in one fused step.
        # ``vocab_size`` sizes them and is zero on an argmax export, which
        # declares neither; the sidecar's own
        # ``runtime_shapes.draft_proposal`` must agree with this record.
        "draft_proposal": draft_proposal,
        "vocab_size": (
            draft_config.vocab_size if draft_proposal == "sampled" else 0
        ),
        # What the frozen export pins off. Logit penalties and pipeline
        # stages are refused by the build itself (a block drafter takes no
        # penalties; this pairing answers no stage speculation), so they hold
        # unconditionally; the other two follow the build.
        "refused": {
            "logit_penalties": True,
            "pipeline_stages": True,
            "in_thinking_phase": not any(
                isinstance(t, TensorType) and t.dtype == DType.bool
                for t in input_types
            ),
            "structured_output": not structured_output,
        },
    }


ADAPTER = TargetAdapter(
    primary_kv=ATTN_CACHE_KEY,
    verify=verify,
    block_proposer=block_proposer,
    export_contract=export_contract,
    # The frozen DSpark input contract is text-only.
    takes_vision_inputs=lambda _config: False,
    # The pairing this fold was written for: K3's accept graph answers its
    # per-step index glue in one launch. Every other target keeps the chain.
    fold_index_glue=True,
)
