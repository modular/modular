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
"""Block spec-decode adapters for a ModuleV3 Mistral Large 4 target.

The target and the draft are one model: the drafter is the checkpoint's own
``mtp_dspark.*`` block, so the fused graph holds both and nothing about the
drafter crosses the ABI. The verify returns the target's variable logits and
the concatenated residual taps the drafter projects, and the proposer drafts a
block of ``depth`` rows over the drafter's own paged latent leaf.

The graph is the block ABI at one attention replica -- TP8 attention beside EP8
routed experts -- so it declares every expert rank's collective buffers
(``ep_tail``) and no data-parallel inputs, and it takes no
``in_thinking_phase`` slot: Mach's Mistral Large 4 block layout is 419 inputs
(397 plus the 22-slot EP tail) with the signal buffers trailing, and a reasoning
parser must not move them (hence ``relaxed_thinking_acceptance=False``).
Mistral Large 4 also refuses logit penalties at admission, so no penalty input
is declared.

The drafter's paged read and write are its own module's
(:meth:`~.dspark.DSparkDrafter.write_context` and
:meth:`~.dspark.DSparkDrafter.forward_block_paged`). The two call sites live in
:meth:`DSparkMistralLarge4Proposer._write_context` and
:meth:`DSparkMistralLarge4Proposer._draft_block`, so the contract between the
proposer and the drafter is one screen long, and a drafter that has not landed
the paged path yet fails loudly there instead of deep inside a trace.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

from max import functional as F
from max.dtype import DType
from max.graph import BufferType, TensorType, TensorValue
from max.nn.common_layers.kv_cache import PagedCacheValues
from max.nn.kv_cache import KVCacheInputsPerDevice
from max.nn.legacy.comm.ep import EPCommBuffers
from max.nn.legacy.transformer import ReturnHiddenStates
from max.pipelines.speculative._tensor_compat import (
    as_graph_values,
    as_tensors,
)
from max.pipelines.speculative.block_driver import (
    Accepted,
    BlockBatch,
    DraftSampler,
)
from max.pipelines.speculative.driver import SequentialBatch, TensorAdapter
from max.pipelines.speculative.spec_adapters.target_adapter import TargetAdapter
from max.pipelines.speculative.spec_target import Verified
from max.tensor import Tensor
from max.tensor.sharding import DeviceMapping, DeviceMesh

from .config import (
    INDEXER_GROUP,
    NOPE_MLA_GROUP,
    ROPE_MLA_GROUP,
    MistralLarge4Config,
)
from .dspark import DSparkDrafter
from .mistral_large4 import MistralLarge4, row_positions

if TYPE_CHECKING:
    from max.pipelines.speculative.block_driver import BlockProposer
    from max.pipelines.speculative.config import SpeculativeConfig
    from max.pipelines.speculative.spec_adapters import VerifyFn

__all__ = [
    "ADAPTER",
    "MASK_TOKEN_ID",
    "DSparkMistralLarge4Proposer",
    "MistralLarge4BlockTarget",
    "block_anchors",
]

_Batch = BlockBatch[Tensor, Tensor, Tensor]
_Caches = list[KVCacheInputsPerDevice[Tensor, Tensor]]
_Shards = list[Tensor]
"""One single-device value per rank, the driver's own currency."""
_Hidden = tuple[_Shards, _Shards]
"""The drafter's block result, per rank: its logits and the tokens it sampled."""
_Block = list[Tensor]
"""What ``embed_block`` hands ``forward_block``: each rank's anchors, then
each rank's embedded block rows.

The driver threads one opaque value between the two phases, so the two per-rank
lists travel packed into one and the block forward splits them again.
"""

MASK_TOKEN_ID = 0
"""The block's fill token, which the drafter never embeds.

Mistral Large 4's drafter has no mask token: the block's rows past the anchor
are the checkpoint's trained ``backbone.mask_embedding`` rows, and the
checkpoint's ``mtp`` block declares no such id. The driver still asks a
proposer for one to fill ``block_ids``' tail, so this is a documented
placeholder drawn from the checkpoint's own vocabulary. Only row 0 of each
block is ever read -- :func:`block_anchors` takes it -- so the value never
reaches an embedding lookup.
``max/tests/integration/architectures/mistral_large4/test_spec_adapters.py``'s
``test_the_mask_token_is_inert`` pins that.
"""


def block_anchors(block_ids: Tensor, block_size: int) -> Tensor:
    """Returns each block's anchor token, ``[batch]``.

    The driver lays a block out request-major, ``block_size`` rows per request,
    and only row 0 of each is a token: the rest are the drafter's trained mask
    rows, which :data:`MASK_TOKEN_ID` fills and nothing reads.

    Args:
        block_ids: ``[batch * block_size]`` the driver's flattened block.
        block_size: Rows per block, the drafter's ``depth``.

    Returns:
        The anchors, in request order.
    """
    return F.reshape(block_ids, [-1, block_size])[:, 0]


class MistralLarge4BlockTarget:
    """The ModuleV3 Mistral Large 4 target over the merged verify rows.

    Acceptance samples its recovered and bonus tokens from these logits, which
    the verify projects over the trailing rows of each request; the taps ride
    back beside them, concatenated in the drafter's column order.
    """

    def __init__(self, target: MistralLarge4, *, vocab_size: int) -> None:
        """Wraps a built target.

        Args:
            target: The target module the fused graph holds.
            vocab_size: The head's padded width.
        """
        self.target = target
        self.vocab_size = vocab_size
        # TP8 attention over one replica takes no data-parallel inputs; the EP
        # collectives ride the adapter's ``ep_tail`` instead.
        self.takes_distributed_inputs = False

    def __call__(
        self, batch: SequentialBatch | BlockBatch
    ) -> Verified[list[TensorValue], TensorValue]:
        """Verifies the batch's blocks against the target."""
        assert isinstance(batch, BlockBatch)
        verified: Verified[list[TensorValue], TensorValue] = as_graph_values(
            self._verify(as_tensors(batch))
        )
        return verified

    def _verify(self, batch: _Batch) -> Verified[list[Tensor], Tensor]:
        """Runs the verify over the merged window, on ``Tensor`` values."""
        comm_buffers: EPCommBuffers | None = None
        ep_batch_manager = self.target.ep_batch_manager
        if ep_batch_manager is not None:
            assert batch.ep_inputs is not None, (
                "a target built with an EP batch manager must get the"
                " graph's EP input buffers"
            )
            comm_buffers = ep_batch_manager.comm_buffers(batch.ep_inputs)
        # The paged-cache kernels take the same cache leaves as graph values,
        # so the buffers they write in place stay buffers through the round
        # trip. A leaf whose group this checkpoint has no owners for is not a
        # cache the target reads, so it is not passed: the target's own
        # ``forward`` and ``verify`` take the same view of the tree.
        kv_by_group: dict[str, list[Any]] = {
            ROPE_MLA_GROUP: as_graph_values(list(batch.kv_collections)),
            INDEXER_GROUP: as_graph_values(
                list(batch.passthrough_kv[INDEXER_GROUP])
            ),
        }
        if NOPE_MLA_GROUP in batch.passthrough_kv:
            kv_by_group[NOPE_MLA_GROUP] = as_graph_values(
                list(batch.passthrough_kv[NOPE_MLA_GROUP])
            )
        logits, _, taps = self.target.verify(
            batch.merged_tokens,
            batch.return_n_logits,
            batch.merged_offsets,
            kv_by_group,
            comm_buffers=comm_buffers,
        )
        # One concatenated tap tensor per rank, replicated over tensor
        # parallelism: the drafter's projection does not shard its columns, so
        # every rank reads every tap.
        hidden = [taps.local_shards[d] for d in range(batch.n_devs)]
        return Verified(logits=logits, hidden=hidden)


class DSparkMistralLarge4Proposer(TensorAdapter):
    """The in-checkpoint DSpark drafter, on its own paged latent leaf.

    The block is ``depth`` rows wide -- the committed anchor, then the
    drafter's learned mask rows -- and the drafter samples every row, so a
    block of three proposes three tokens and the target verifies four. Its
    context is the target's taps for every verified row, written into the
    drafter's per-layer latent cache at the pre-iteration lengths; the block
    then attends over that window plus its own rows.

    This drafter embeds inside its paged forward -- it borrows the target's
    token embedding and LM head as call arguments rather than as parameters --
    so :meth:`embed_block` runs the target's own embedding module over the
    anchors and :meth:`forward_block` appends the drafter's trained mask rows and
    drafts. Both borrowed tables are vocabulary-sharded over tensor
    parallelism, which is why neither is a weight the drafter could read
    itself.
    """

    samples_from_anchor = True
    supports_zero_draft_rows = False
    # The drafter's own cache lengths place its context: the write lands in the
    # drafter's leaf, at the lengths that leaf carries.
    ctx_at_draft_cache_length = True

    def __init__(
        self,
        drafter: DSparkDrafter,
        target: MistralLarge4,
        *,
        block_size: int,
        mask_token_id: int = MASK_TOKEN_ID,
    ) -> None:
        """Binds the drafter to the target whose embedding and head it reads.

        Args:
            drafter: The target checkpoint's ``mtp_dspark.*`` block.
            target: The target module, which lends its token embedding and LM
                head as call arguments.
            block_size: The drafter's trained block width, ``depth``.
            mask_token_id: The block's fill token (see :data:`MASK_TOKEN_ID`).

        Raises:
            ValueError: If the target has no mesh to place the taps on.
        """
        self.drafter = drafter
        self.target = target
        self.block_size = block_size
        self.mask_token_id = mask_token_id
        mesh = target.config.mesh
        if mesh is None:
            raise ValueError(
                "a Mistral Large 4 DSpark proposer needs the target's mesh"
            )
        self.mesh: DeviceMesh = mesh

    @property
    def config(self) -> MistralLarge4Config:
        """The target's config, which carries the drafter's taps."""
        return self.target.config

    def materialize(
        self,
        batch: _Batch,
        target_hidden: list[Tensor],
        ctx_kv: _Caches,
    ) -> None:
        """Writes the target's taps for every verified row into the leaf."""
        ctx = self._leaf(ctx_kv)
        # The driver's merged batch is one request-major tensor on the first
        # device; every rank holds the same rows, so both stage onto the mesh
        # before the per-row position builder reads them.
        tokens = self._broadcast(batch.merged_tokens)
        row_offsets = self._broadcast(batch.merged_offsets)
        positions = row_positions(tokens, row_offsets, ctx.cache_lengths)
        if self.mesh.num_devices > 1:
            # Every rank holds every merged row, so each row's position is the
            # same on every rank. Claim that placement: the write's per-row
            # tables are declared replicated beside it.
            positions = positions.rebind_mapping(
                DeviceMapping.replicated(self.mesh)
            )
        self._write_context(
            self._replicated(target_hidden),
            positions,
            row_offsets,
            ctx,
        )

    def embed_block(self, batch: _Batch, block_ids: Tensor) -> _Block:
        """Embeds each block's anchor through the target's own embedding.

        The target shards its vocabulary across the tensor-parallel ranks, so
        the drafter cannot look a token up in the weight itself: the
        embedding module gathers each rank's slice and all-reduces, which is
        the collective the target's own first layer runs. The block's mask rows
        are the drafter's trained parameters and are appended after this, in
        the block forward.

        Returns:
            The block's anchor ids and its block rows ``[batch * depth,
            hidden]``, each one value per device, packed as :data:`_Block`.
        """
        del batch
        anchors = self._broadcast(block_anchors(block_ids, self.block_size))
        embeds = self.target.language_model.tok_embeddings(
            anchors.cast(DType.int32)
        )
        block = self.drafter.block_from_anchors(embeds)
        return [*self._shards(anchors), *self._shards(block)]

    def forward_block(
        self,
        batch: _Batch,
        embeds: _Block,
        offsets: list[Tensor],
        block_kv: _Caches,
    ) -> _Hidden:
        """Runs the windowed block over the drafter's leaf.

        ``offsets`` is unread: the block's rows sit at fixed ``block_size``
        strides after the anchors, whose own positions the leaf's cache lengths
        give.
        """
        del offsets
        # Every rank's anchors, then every rank's rows (see ``_Block``).
        ranks = batch.n_devs
        if len(embeds) != 2 * ranks:
            raise ValueError(
                "the packed block is one value per rank for the anchors and"
                f" one for the rows: got {len(embeds)} for {ranks} ranks"
            )
        anchors = self._replicated(embeds[:ranks])
        block = self._leaf(block_kv)
        # The block cache's lengths are the pre-iteration length plus what this
        # iteration commits, so the anchor -- the token it commits -- is one
        # below that length.
        positions = F.cast(block.cache_lengths, DType.int32) - 1
        logits, tokens = self._draft_block(
            anchors, self._replicated(embeds[ranks:]), positions, block
        )
        # The driver carries one single-device value per rank, so a mesh
        # result leaves as its own shards: every rank holds the whole block,
        # so they are the same rows on each.
        return self._shards(logits), self._shards(tokens)

    def head(
        self,
        batch: _Batch,
        block_hs: _Hidden,
        accepted: Accepted,
        sampler: DraftSampler,
    ) -> Tensor:
        """Returns the drafter's own draft tokens, one per block row."""
        del accepted, sampler
        _, tokens = block_hs
        # The driver's outputs are one device's graph values, and every rank
        # holds the same block, so rank 0's copy is the block. Its shape is
        # already request-major -- the block's own depth is the trailing axis --
        # so the row count only needs naming as the driver's sequence count.
        return tokens[0].reshape([batch.num_draft_seqs, self.block_size])

    def _write_context(
        self,
        taps: Tensor,
        positions: Tensor,
        row_offsets: Tensor,
        kv: PagedCacheValues,
    ) -> None:
        """Writes the context latents into the drafter's leaf.

        The one call into the drafter's paged write path
        (``DSparkDrafter.write_context``), kept here so a change to that
        contract is one line on this side. The rows land at the leaf's own
        cache lengths -- the pre-iteration lengths, since the context is
        written before this iteration's block.

        Args:
            taps: ``[rows, n_taps * hidden]`` the target's taps of the merged
                verify rows.
            positions: ``[rows]`` each row's absolute position.
            row_offsets: ``[batch + 1]`` the merged ragged offsets.
            kv: The drafter's leaf, at its pre-iteration cache lengths.
        """
        self.drafter.write_context(taps, positions, row_offsets, kv)

    def _draft_block(
        self,
        anchors: Tensor,
        block_rows: Tensor,
        anchor_positions: Tensor,
        kv: PagedCacheValues,
    ) -> tuple[Tensor, Tensor]:
        """Runs the paged block forward and returns its logits and tokens.

        ``DSparkDrafter.forward_block_paged`` samples its own rows (greedy
        argmax, the serving graph's route), which is why the proposer's head
        hands its tokens through rather than scoring them again.

        Args:
            anchors: ``[batch]`` each block's anchor token.
            block_rows: ``[batch * depth, hidden]`` the block, its anchors
                already embedded through the target's embedding.
            anchor_positions: ``[batch]`` each block's anchor position, the
                last token this iteration commits; its own row is the window's
                last key.
            kv: The drafter's leaf at its block position, its dispatch metadata
                rebuilt at the block's width.

        Returns:
            The block's ``[batch, depth, vocab]`` logits and its
            ``[batch, depth]`` tokens.
        """
        return self.drafter.forward_block_paged(
            block_rows,
            anchors,
            anchor_positions,
            kv,
            self._lm_head(),
        )

    def _leaf(self, caches: _Caches) -> PagedCacheValues:
        """The drafter's cache as one mesh tensor set.

        Only the paged fields the drafter's own math reads travel: its
        attention is hand-rolled over gathered rows, so nothing consumes the
        dispatch record and this graph leaves both declared dispatch slots --
        the MLA record at the verify width and the ``draft_`` copy -- to the
        engine, which stages them like any other ABI slot. That is the same
        position the other block exports take when they rebuild the record
        in-graph instead of reading it.
        """
        return PagedCacheValues.from_upstream(
            as_graph_values(list(caches)),
            DeviceMapping.replicated(self.mesh),
        )

    def _replicated(self, values: Sequence[Tensor]) -> Tensor:
        """One replicated tensor from each device's copy."""
        return Tensor.from_shard_values(
            list(values), DeviceMapping.replicated(self.mesh)
        )

    def _shards(self, value: Tensor) -> _Shards:
        """A mesh tensor's per-rank values, which the driver carries."""
        if not value.is_distributed:
            return [value]
        return list(value.local_shards)

    def _broadcast(self, value: Tensor) -> Tensor:
        """Places a single-device tensor on every device of the mesh."""
        if self.mesh.num_devices > 1:
            return F.distributed_broadcast(value, self.mesh)
        return value.to(self.mesh)

    def _lm_head(self) -> Callable[[Tensor], Tensor]:
        """The target's LM head, which the drafter scores its block with.

        It shards its output vocabulary across the tensor-parallel ranks, so
        one call returns one rank's columns: the drafter all-gathers them back.
        """
        return self.target.language_model.lm_head


def verify(
    target: Any,
    target_config: Any,
    hidden_states: ReturnHiddenStates,
    draft_config: Any,
) -> VerifyFn:
    """Returns the verify of a ModuleV3 Mistral Large 4 target.

    Raises:
        ValueError: If the target wasn't built to return selected layers, which
            only block drafts read.
    """
    del draft_config
    if hidden_states != ReturnHiddenStates.SELECTED_LAYERS:
        raise ValueError(
            "A Mistral Large 4 target verifies only for block drafts, which"
            f" read selected layers. Got {hidden_states.value}."
        )
    assert isinstance(target, MistralLarge4)
    return MistralLarge4BlockTarget(
        target, vocab_size=int(target_config.vocab_size)
    )


def ep_input_types(target: Any) -> Sequence[TensorType | BufferType]:
    """Returns the EP buffers a target built with expert parallelism takes."""
    assert isinstance(target, MistralLarge4)
    if target.ep_batch_manager is None:
        return ()
    return tuple(target.ep_batch_manager.input_types())


def block_proposer(
    verify: VerifyFn,
    draft: Any,
    draft_config: Any,
    speculative: SpeculativeConfig,
) -> BlockProposer[Any, Any, Any, Any, Any]:
    """Returns the DSpark proposer, which reads the target's embedding and head."""
    del speculative
    assert isinstance(verify, MistralLarge4BlockTarget)
    assert isinstance(draft, DSparkDrafter)
    block_size = getattr(draft_config, "block_size", None)
    if block_size is None:
        raise ValueError(
            "the Mistral Large 4 DSpark draft config records no block size"
        )
    return DSparkMistralLarge4Proposer(
        draft,
        verify.target,
        block_size=int(block_size),
    )


def check(speculative: SpeculativeConfig, structured_output: bool) -> None:
    """Refuses a speculative config this graph cannot serve.

    The graph drafts by argmax: the drafter's paged block forward picks every
    row's token through its own recurrent chain, which is where a sampled
    route's per-position distributions would have to come from. Logit
    penalties never reach a block draft at all -- the driver refuses them for
    every block export.

    Raises:
        ValueError: If the config asks for a sampled draft proposal.
    """
    del structured_output
    if speculative.draft_proposal != "argmax":
        raise ValueError(
            "Mistral Large 4's DSpark graph drafts by argmax only: the"
            " drafter's paged block forward emits its own tokens, not the"
            " per-position distributions --draft-proposal"
            f" {speculative.draft_proposal} would draw from. Drop"
            " --draft-proposal."
        )


ADAPTER = TargetAdapter(
    # The rope layers' latent leaf, which the driver dispatches the target over.
    primary_kv=ROPE_MLA_GROUP,
    verify=verify,
    block_proposer=block_proposer,
    ep_input_types=ep_input_types,
    # TP8 attention beside EP8 routed experts is one attention replica, so the
    # graph declares the EP collective buffers on an otherwise TP-only
    # signature: after the KV tree, with none of the data-parallel inputs.
    ep_tail=lambda target: target.ep_batch_manager is not None,
    check=check,
    # Mach's Mistral Large 4 block layout has no in_thinking_phase slot: the
    # graph is 419 inputs (397 plus the 22-slot EP tail) with the signal
    # buffers trailing, and the relaxed in-thinking rule would add one more.
    relaxed_thinking_acceptance=False,
    # The trace appends the mesh's signal buffers after every declared input,
    # which is the layout's Trailing placement on every device count.
    signal_buffer_index=None,
)
