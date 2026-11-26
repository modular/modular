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
"""Inkling 2's verify, and the DSpark block drafter its checkpoint ships.

The verify runs the whole decoder over the merged window and returns the
eight tapped hidden states the drafter reads, with the rows' offsets.

The conv state needs no rollback. It is a ring of each sequence's recent
inputs indexed by position, sized ``kernel_size - 1 + num_draft_tokens`` by
:class:`~.state_cache.ConvStateLayout`. A verify writes the committed token
and the drafts at their positions; the next step reads only the
``kernel_size - 1`` positions below its committed length, all of which this
step writes as accepted tokens or never reaches, and it starts writing at
the committed length, so every rejected position is overwritten before any
kernel reads it. The KV pages go without a rewind by the same argument.

The drafter has two caches of its own: its full layers' and its local
layers'. The driver drives the first and hands the second through
``passthrough_kv`` at the pre-iteration lengths, so each phase restates its
lengths to match the driven one. Its heads shard over the target's
tensor-parallel axis, as the target's do, because the caches share the
target's pool.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any

from max import functional as F
from max.graph import DeviceRef, TensorValue
from max.nn.common_layers.kv_cache import PagedCacheValues
from max.nn.kv_cache import RecurrentStateInputsPerDevice
from max.nn.outputs import ReturnHiddenStates
from max.pipelines.speculative._tensor_compat import (
    as_graph_values,
    as_tensors,
)
from max.pipelines.speculative.block_driver import (
    Accepted,
    ArgmaxDraftSampler,
    BlockBatch,
    DraftSampler,
    block_kv_with_dispatch,
)
from max.pipelines.speculative.config import SpeculativeConfig
from max.pipelines.speculative.driver import SequentialBatch
from max.pipelines.speculative.ragged_token_merger import shape_to_scalar
from max.pipelines.speculative.spec_adapters.target_adapter import (
    TargetAdapter,
)
from max.pipelines.speculative.spec_target import Verified
from max.tensor import Tensor
from max.tensor.sharding import DeviceMapping, DeviceMesh

from .dspark import DSPARK_GLOBAL_ATTENTION, DSPARK_LOCAL_ATTENTION
from .dspark_draft import (
    MASK_TOKEN_ID,
    Inkling2DSparkDraft,
    Inkling2DSparkDraftConfig,
)
from .inkling2 import Inkling2
from .model_config import GLOBAL_ATTENTION, LOCAL_ATTENTION, STATE_CACHE_KEY
from .modules import Inkling2Decoder

__all__ = ["ADAPTER", "Inkling2BlockTarget", "Inkling2DSparkProposer"]

_DRAFT_LOCAL = f"draft_{DSPARK_LOCAL_ATTENTION}"
"""The drafter's local cache under ``passthrough_kv``."""

_shape_to_scalar = F.functional(shape_to_scalar)

Taps = tuple[Tensor, Tensor]
"""The verify payload: every row's eight tapped states concatenated,
``[rows, 8 * hidden]``, and the merged window's row offsets."""


def _replicate(t: Tensor, mesh: DeviceMesh) -> Tensor:
    """Replicates a device-zero input across the mesh, as the target does."""
    if mesh.num_devices > 1:
        return F.distributed_broadcast(t, mesh)
    return t.to(mesh)


def _paged(leaves: Any, mesh: DeviceMesh) -> PagedCacheValues:
    return PagedCacheValues.from_upstream(
        list(leaves), DeviceMapping.replicated(mesh)
    )


def _sample_next(sampler: DraftSampler) -> Callable[[Tensor, int], Tensor]:
    """Bridges the driver's per-position sampler into the chain's world.

    The framework's sampler protocol spells its inputs as graph values while
    the Markov chain runs on mesh ``Tensor``s, so the logits wrap on the way
    in and the pick unwraps on the way out.
    """

    def sample(step_logits: Tensor, position: int) -> Tensor:
        picked = sampler.sample_next(TensorValue(step_logits), position)
        return Tensor.from_graph_value(picked)

    return sample


class Inkling2BlockTarget:
    """Verifies the previous drafts with the decoder over the merged window.

    Every row of the window runs through the decoder, so its KV and conv
    inputs are written whatever the accept later commits; rows past the
    committed length are overwritten by the next verify before any read.
    """

    def __init__(self, model: Inkling2, taps: tuple[int, ...]) -> None:
        self.model = model
        self.taps = taps

    def __call__(self, batch: SequentialBatch | BlockBatch) -> Verified[Any]:
        """Runs the decoder over the merged window and returns its taps."""
        assert isinstance(batch, BlockBatch)
        b = as_tensors(batch)
        model = self.model
        conv: list[RecurrentStateInputsPerDevice[Any, Any]] = []
        for device in b.passthrough_kv[STATE_CACHE_KEY]:
            assert isinstance(device, RecurrentStateInputsPerDevice)
            conv.append(device)
        out = model.run_layers_on(
            b.merged_tokens,
            b.merged_offsets,
            None,
            {
                GLOBAL_ATTENTION: b.kv_collections,
                LOCAL_ATTENTION: b.passthrough_kv[LOCAL_ATTENTION],
            },
            conv,
            self.taps,
        )
        # The drafter's taps end with the final hidden state of every row,
        # which the head would otherwise mix again for the window's rows.
        final_hidden = (
            out.taps[-1]
            if model.text_config.num_hidden_layers in self.taps
            else None
        )
        logits = model.window_logits(
            out.blocks,
            out.h,
            out.input_row_offsets,
            b.return_n_logits,
            final_hidden=final_hidden,
        )
        payload: Taps = (F.concat(out.taps, axis=-1), out.input_row_offsets)
        return Verified(logits=as_graph_values(logits), hidden=payload)


class Inkling2DSparkProposer:
    """Drafts ``block_size - 1`` tokens per sequence from one block forward.

    The block is the committed token then mask rows; the drafter embeds the
    anchor through the target's table and the mask rows through its own
    ``mask_embed``, so the mask token id fills the ids only to give the block
    its shape. The anchor's own row proposes nothing.
    """

    samples_from_anchor = False
    supports_zero_draft_rows = False

    def __init__(self, drafter: Inkling2DSparkDraft, model: Inkling2) -> None:
        self.drafter = drafter.dspark
        self.model = model
        self.mesh: DeviceMesh = model.norm.weight.mesh
        self.block_size = self.drafter.config.block_size
        self.mask_token_id = MASK_TOKEN_ID
        self._taps: Taps | None = None

    def _with_local(
        self, batch: BlockBatch, driven: list[Any]
    ) -> dict[str, PagedCacheValues]:
        """Both draft caches, the local one at the driven one's lengths.

        A checkpoint that names no local layer builds no local child, so the
        mapping carries the driven cache alone then.
        """
        caches = {
            DSPARK_GLOBAL_ATTENTION: _paged(as_tensors(driven), self.mesh)
        }
        local = batch.passthrough_kv.get(_DRAFT_LOCAL)
        if local:
            caches[DSPARK_LOCAL_ATTENTION] = _paged(
                as_tensors(
                    [
                        replace(leaf, cache_lengths=kv.cache_lengths)
                        for leaf, kv in zip(local, driven, strict=True)
                    ]
                ),
                self.mesh,
            )
        return caches

    def materialize(
        self, batch: BlockBatch, target_hidden: Taps, ctx_kv: list[Any]
    ) -> None:
        """Writes every drafter layer's context K and V for the window."""
        taps, row_offsets = target_hidden
        self._taps = target_hidden
        self.drafter.materialize(
            taps, self._with_local(batch, ctx_kv), row_offsets
        )

    def embed_block(
        self, batch: BlockBatch, block_ids: TensorValue
    ) -> list[TensorValue]:
        """Hands the ids through; the block embeds in :meth:`forward_block`."""
        return [block_ids]

    def forward_block(
        self,
        batch: BlockBatch,
        embeds: list[TensorValue],
        offsets: list[TensorValue],
        block_kv: list[Any],
    ) -> Tensor:
        """Runs the drafter's layers over each sequence's block.

        The block's caches are rebuilt at the block's own query width: the
        driver hands over the verify's, and a prefill batch's wider query
        drives the first block attention layer to NaN.
        """
        ids = _replicate(as_tensors(embeds[0]), self.mesh)
        k = self.block_size
        anchors = ids.reshape([-1, k])[:, 0]
        model = self.model
        anchor_embeds = model.embed_norm(model.embed(anchors))
        block = self.drafter.embed_block(anchor_embeds)
        block_offsets = _replicate(as_tensors(offsets[0]), self.mesh)
        return self.drafter.forward(
            block,
            self._with_local(batch, block_kv_with_dispatch(list(block_kv), k)),
            block_offsets,
        )

    def head(
        self,
        batch: BlockBatch,
        block_hs: Tensor,
        accepted: Accepted,
        sampler: DraftSampler,
    ) -> TensorValue:
        """Drafts the block's mask rows through the Markov head.

        A sampled proposal draws each step through the driver's sampler, so
        the distributions the accept phase reads are the ones it drew from.
        """
        assert self._taps is not None, "head runs after materialize"
        taps, _ = self._taps
        b = as_tensors(batch)
        # The Markov head's context is the taps of each sequence's last
        # accepted row: the window's last row, less the drafts it rejected.
        # The indices build on device zero, where the accept ran.
        last = F.rebind(b.merged_offsets[1:] - 1, ["batch_size"])
        drafts = _shape_to_scalar(
            b.num_draft_tokens,
            DeviceRef.from_device(last.device),
            last.dtype,
        )
        accepted_rows = (
            last - drafts + as_tensors(accepted.num_accepted).cast(last.dtype)
        )
        rows = _replicate(accepted_rows, self.mesh)
        f_rnn = self.drafter.f_rnn(F.gather(taps, rows, axis=0))
        anchors = _replicate(as_tensors(accepted.next_tokens), self.mesh)
        text = self.model.text_config
        drafts = self.drafter.draft(
            as_tensors(block_hs),
            anchors,
            f_rnn,
            self.model.unembed.weight,
            final_logit_softcapping=text.final_logit_softcapping,
            unpadded_vocab_size=text.unpadded_vocab_size,
            sampler=None
            if type(sampler) is ArgmaxDraftSampler
            else _sample_next(sampler),
        )
        return as_graph_values(drafts.local_shards[0])


def verify(
    target: Any,
    target_config: Any,
    hidden_states: ReturnHiddenStates,
    draft_config: Any,
) -> Inkling2BlockTarget:
    """Returns Inkling 2's verify, which returns the DSpark taps.

    Raises:
        ValueError: If the pairing doesn't read selected layers, as only the
            DSpark block draft does.
    """
    del target_config
    if hidden_states != ReturnHiddenStates.SELECTED_LAYERS:
        raise ValueError(
            "An Inkling 2 target verifies only for its DSpark block draft,"
            f" which reads selected layers. Got {hidden_states.value}."
        )
    assert isinstance(target, Inkling2Decoder)
    assert isinstance(draft_config, Inkling2DSparkDraftConfig)
    assert draft_config.dspark is not None
    return Inkling2BlockTarget(
        target.model, tuple(draft_config.dspark.target_layer_ids)
    )


def block_proposer(
    verify: Any,
    draft: Any,
    draft_config: Any,
    speculative: SpeculativeConfig,
) -> Inkling2DSparkProposer:
    """Returns the DSpark proposer, on the verify's decoder."""
    del draft_config, speculative
    assert isinstance(verify, Inkling2BlockTarget)
    assert isinstance(draft, Inkling2DSparkDraft)
    return Inkling2DSparkProposer(draft, verify.model)


ADAPTER = TargetAdapter(
    primary_kv=GLOBAL_ATTENTION,
    verify=verify,
    block_proposer=block_proposer,
    # The fused graph takes no in-thinking input.
    relaxed_thinking_acceptance=False,
)
