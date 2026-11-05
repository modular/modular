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
"""GLM-5.3-Flash's call on ModuleV3, and the MTP layer its checkpoint ships.

The verify and its rollback are the graph-API model's (see
:mod:`~max.pipelines.architectures.glm5_next.spec_adapters`): verifying K draft
tokens advances 34 KDA recurrences and every sparse layer's indexer tail ring,
none of which a length pointer rewinds, so the verify defers all of them and
:class:`~.spec_state.Glm5NextSpecState` rolls them onto the accepted prefix.
The verify runs on graph values, as the driver hands them; the proposer runs on
``Tensor`` values, as the ModuleV3 draft takes them.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import partial
from typing import TYPE_CHECKING, Any

from max import functional as F
from max.driver import CPU, Buffer
from max.dtype import DType
from max.graph import BufferType, BufferValue, DimLike, TensorType, TensorValue
from max.nn.common_layers.kv_cache import PagedCacheValues
from max.nn.kv_cache import RecurrentStateInputsPerDevice
from max.nn.legacy.comm.ep import EPCommBuffers
from max.nn.outputs import ReturnHiddenStates
from max.pipelines.architectures.glm5_next.state_cache import (
    INDEXER_CACHE_KEY,
    MLA_CACHE_KEY,
    STATE_CACHE_KEY,
    TAIL_RING_LEAF_ID,
)
from max.pipelines.speculative._tensor_compat import as_graph_values
from max.pipelines.speculative.block_driver import BlockBatch
from max.pipelines.speculative.config import SpeculativeConfig
from max.pipelines.speculative.driver import (
    CarryDimNames,
    DecodeKVSwap,
    DraftCache,
    DraftStepInput,
    Proposed,
    ReuseSpec,
    SequentialBatch,
    TensorAdapter,
    gather_variable_rows,
)
from max.pipelines.speculative.spec_adapters.target_adapter import TargetAdapter
from max.pipelines.speculative.spec_target import DistributedVerify, Verified
from max.tensor import Tensor
from max.tensor.sharding import DeviceMapping, DeviceMesh

from ..deepseekV3_2_modulev3.deepseekV3_2 import project_to_logits
from ..deepseekV3_modulev3.deepseekV3 import gather_last_tokens
from .glm5_next import Glm5Next, state_leaf
from .layers.indexer import KPoolTail, PooledIndexer
from .modules import Glm5NextDecoder
from .mtp import Glm5NextMTP
from .spec_state import Glm5NextSpecState

if TYPE_CHECKING:
    from max.pipelines.speculative.driver import SequentialProposer
    from max.pipelines.speculative.spec_adapters import VerifyFn

__all__ = ["ADAPTER", "Glm5NextModuleV3MTPProposer"]

_Batch = SequentialBatch[Tensor, Tensor, Tensor]


def check(speculative: SpeculativeConfig, structured_output: bool) -> None:
    """Raises if a GLM-5.3-Flash target can't verify under the config.

    Raises:
        ValueError: If acceptance is synthetic.
    """
    del structured_output
    if speculative.synthetic_acceptance_rate is not None:
        raise ValueError(
            "synthetic acceptance would bypass the KDA state rollback's"
            " accepted-length plan"
        )


def _replicate(value: Tensor, mesh: DeviceMesh) -> Tensor:
    if mesh.num_devices > 1:
        return F.distributed_broadcast(value, mesh)
    # The broadcast collective needs signal buffers, which only a
    # multi-device graph has. Onto one device, replicating is a placement.
    return value.to(mesh)


def _state(
    batch: SequentialBatch[Any, Any, Any],
) -> list[RecurrentStateInputsPerDevice[Any, Any]]:
    """Returns the per-device state leaves the batch carries."""
    state: list[RecurrentStateInputsPerDevice[Any, Any]] = []
    for device in batch.passthrough_kv[STATE_CACHE_KEY]:
        assert isinstance(device, RecurrentStateInputsPerDevice)
        state.append(device)
    return state


def verify(
    target: Any,
    target_config: Any,
    hidden_states: ReturnHiddenStates,
    draft_config: Any,
) -> VerifyFn:
    """Returns the verify of a ModuleV3 GLM-5.3-Flash model.

    It records its KDA state's updates in the verify ring, defers its
    indexers' tail seeds, and rolls both onto the accepted prefix.

    Raises:
        ValueError: If the target wasn't built to return its last normalized
            layer, which the MTP layer reads.
    """
    del target_config, draft_config
    target = _language_model(target)
    if hidden_states != ReturnHiddenStates.ALL_NORMALIZED:
        raise ValueError(
            "A GLM-5.3-Flash target verifies only for drafts that read its"
            f" last normalized layer. Got {hidden_states.value}."
        )
    return DistributedVerify(_verify, Glm5NextSpecState(target))


def _language_model(target: Any) -> Glm5Next:
    """Returns the model a built target runs.

    The registered decoder wraps it, under the same weight names.
    """
    if isinstance(target, Glm5NextDecoder):
        target = target.model
    assert isinstance(target, Glm5Next)
    return target


def _verify(
    state: Glm5NextSpecState, batch: SequentialBatch | BlockBatch
) -> Verified[list[TensorValue]]:
    assert isinstance(batch, SequentialBatch)
    language_model = state.language_model
    mesh = language_model.mesh
    assert mesh is not None
    replicated = DeviceMapping.replicated(mesh)
    comm_buffers: EPCommBuffers | None = None
    ep_batch_manager = state.target.ep_batch_manager
    if ep_batch_manager is not None:
        assert batch.ep_inputs is not None
        comm_buffers = ep_batch_manager.comm_buffers(batch.ep_inputs)

    offsets = Tensor.from_shard_values(batch.merged_offsets_per_dev, replicated)
    state_inputs = _state(batch)
    with state.capturing(
        state_inputs,
        batch.num_draft_tokens,
        batch.merged_offsets_per_dev,
        replicated,
    ):
        h = language_model.hidden_states(
            _replicate(Tensor.from_graph_value(batch.merged_tokens), mesh),
            offsets,
            PagedCacheValues.from_upstream(batch.kv_collections, replicated),
            PagedCacheValues.from_upstream(
                batch.passthrough_kv[INDEXER_CACHE_KEY], replicated
            ),
            state_inputs,
            comm_buffers,
            subgraphs=False,
        )
    # The verify reads only the logits over its window, so it skips the
    # last-token projection, a vocab-wide matmul per step.
    rows, _ = gather_variable_rows(
        h, offsets, Tensor.from_graph_value(batch.return_n_logits), mesh
    )
    logits = project_to_logits(
        rows,
        language_model.norm,
        language_model.lm_head,
        mesh,
        language_model.config.vocab_size,
        None,
    )
    hidden = language_model.norm(h)
    return Verified(
        logits=as_graph_values(logits),
        hidden=[TensorValue(shard) for shard in hidden.local_shards],
        roll_forward=partial(_roll_forward, state),
    )


def _roll_forward(state: Glm5NextSpecState, batch: SequentialBatch) -> None:
    assert batch.num_accepted is not None
    state.roll_forward(
        _state(batch),
        merged_offsets=batch.merged_offsets,
        merged_offsets_per_dev=batch.merged_offsets_per_dev,
        num_accepted=batch.num_accepted,
        signal_buffers=_signal_buffers(batch.signal_buffers),
        device=batch.device0,
    )


def _signal_buffers(values: list[Any]) -> list[BufferValue]:
    """Returns the batch's signal buffers as graph buffers.

    The rollback's broadcasts take them.
    """
    buffers = as_graph_values(list(values))
    assert all(isinstance(b, BufferValue) for b in buffers)
    return buffers


def sequential_proposer(
    verify: VerifyFn,
    draft: Any,
    draft_config: Any,
    speculative: SpeculativeConfig,
) -> SequentialProposer[Any, Any, Any, Any]:
    """Returns the proposer of the MTP layer, beside the verify's state."""
    del draft_config, speculative
    assert isinstance(verify, DistributedVerify)
    (state,) = verify.args
    assert isinstance(state, Glm5NextSpecState)
    assert isinstance(draft, Glm5NextMTP)
    return Glm5NextModuleV3MTPProposer(draft, state)


class Glm5NextModuleV3MTPProposer(TensorAdapter):
    """The MTP draft, projecting through the target's ``lm_head``.

    Step 0 runs the draft's own indexer over the verified window; every later
    step reuses that selection with its lag. So only step 0 touches the
    draft's tail ring, and it is seeded with the accepted rows once that step
    has run.
    """

    decode_swaps: tuple[DecodeKVSwap, ...] = (
        DecodeKVSwap.MAX_PROMPT_LENGTH_ONE,
        DecodeKVSwap.DRAFT_ATTENTION_DISPATCH_METADATA,
        DecodeKVSwap.DRAFT_MLA_NUM_PARTITIONS,
    )
    passthrough_decode_swaps: tuple[str, ...] = ()
    draft_cache = DraftCache.OWN
    split_prefix = "mtp"
    carry_dim_names = CarryDimNames(prefix="mtp_step", per_device=True)
    step_hidden_mode = ReturnHiddenStates.LAST_PER_DEVICE
    uses_thinking_phase = True
    supports_zero_draft_rows = False
    takes_distributed_inputs = True

    def __init__(self, draft: Glm5NextMTP, state: Glm5NextSpecState) -> None:
        self.draft = draft
        self.state = state
        self.language_model = state.language_model
        mesh = self.language_model.mesh
        assert mesh is not None
        self.mesh = mesh
        self.hidden_dim: DimLike = draft.config.hidden_size
        self.reuse: ReuseSpec | None = ReuseSpec(
            dim=draft.selection_width, split_prefix="mtp_topk"
        )

    @property
    def _replicated(self) -> DeviceMapping:
        return DeviceMapping.replicated(self.mesh)

    def prefill(
        self, batch: _Batch, tokens: Tensor, target_hidden: list[Tensor]
    ) -> Proposed[Tensor]:
        """Drafts the first token from the verified window."""
        indexer = self.draft.indexer
        assert isinstance(indexer, PooledIndexer)
        with self.state.capturing_draft(
            indexer,
            as_graph_values(list(batch.merged_offsets_per_dev)),
            self._replicated,
        ):
            h, selection, offsets = self._decode(
                batch, tokens, target_hidden, None
            )
        self.state.replay_draft_tail(_signal_buffers(batch.signal_buffers))
        # Every row's hidden state and the logits of each request's last
        # ``return_n_logits`` rows, which the driver gathers at the accepted
        # position.
        rows, _ = gather_variable_rows(
            h, offsets, batch.return_n_logits, self.mesh
        )
        return Proposed(
            logits=self._project(rows),
            hidden=list(h.local_shards),
            reuse=list(selection.local_shards),
        )

    def step(
        self, batch: _Batch, draft_input: DraftStepInput[Tensor], index: int
    ) -> Proposed[Tensor]:
        """Drafts the next token from the previous step's."""
        # ``hnorm`` expects the post-``shared_head_norm`` hidden state, which
        # is row-wise, so normalizing the one carried row here is equivalent.
        hidden = self.draft.shared_head_norm(
            Tensor.from_shard_values(draft_input.hidden, self._replicated)
        )
        # Every step reuses step 0's selection, made for the token one
        # position before step 1's.
        h, _, offsets = self._decode(
            batch,
            draft_input.tokens,
            hidden,
            draft_input.reuse,
            selection_lag=index,
        )
        last = gather_last_tokens(h, offsets)
        return Proposed(
            logits=self._project(last), hidden=list(last.local_shards)
        )

    def _project(self, h: Tensor) -> Tensor:
        return project_to_logits(
            h,
            self.draft.shared_head_norm,
            self.language_model.lm_head,
            self.mesh,
            self.draft.config.vocab_size,
            None,
        )

    def _decode(
        self,
        batch: _Batch,
        tokens: Tensor,
        hidden: list[Tensor] | Tensor,
        reuse: list[Tensor] | None,
        selection_lag: int = 0,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Runs the draft layer.

        Returns:
            Its output, its token selection and the query offsets.
        """
        replicated = self._replicated
        embedded = self.language_model.embed_tokens(
            _replicate(tokens, self.mesh)
        )
        if isinstance(hidden, list):
            hidden = Tensor.from_shard_values(hidden, replicated)
        offsets = Tensor.from_shard_values(
            batch.query_offsets_per_dev, replicated
        )
        tail_pool, tail_rows = state_leaf(
            _state(batch), TAIL_RING_LEAF_ID, replicated, head_sharded=False
        )
        comm_buffers: EPCommBuffers | None = None
        if self.draft.ep_batch_manager is not None:
            assert batch.ep_inputs is not None
            comm_buffers = self.draft.ep_batch_manager.comm_buffers(
                batch.ep_inputs
            )
        h, selection = self.draft(
            embedded,
            hidden,
            PagedCacheValues.from_upstream(
                batch.draft_kv_collections, replicated
            ),
            PagedCacheValues.from_upstream(
                batch.passthrough_kv[INDEXER_CACHE_KEY], replicated
            ),
            KPoolTail(
                pool=tail_pool,
                rows=tail_rows,
                layer=F.constant(
                    self.draft.cache_layer_idx, DType.int64, device=CPU()
                ),
            ),
            offsets,
            F.cast(self.language_model.rope.freqs_cis, embedded.dtype).to(
                self.mesh
            ),
            comm_buffers,
            Tensor.from_shard_values(reuse, replicated)
            if reuse is not None
            else None,
            selection_lag,
        )
        return h, selection, offsets


def ep_input_types(target: Any) -> Sequence[TensorType | BufferType]:
    """Returns the expert-parallel inputs of a built target module."""
    target = _language_model(target)
    if target.ep_batch_manager is None:
        return ()
    return target.ep_batch_manager.input_types()


def batch_distributed_buffers(
    target_inputs: object,
) -> tuple[Sequence[Buffer] | None, Sequence[Buffer]]:
    """Returns no context lengths, and the target's EP buffers.

    The EP buffers are the ones the target's batching made.
    The target's batching keeps one context length per data-parallel
    replica, where the distributed signature takes one per device. The fused
    graph never reads them, since every sparse layer runs the decode kernel,
    so the pairing's batching fills them.
    """
    return None, tuple(getattr(target_inputs, "ep_inputs", None) or ())


ADAPTER = TargetAdapter(
    primary_kv=MLA_CACHE_KEY,
    verify=verify,
    ep_input_types=ep_input_types,
    batch_distributed_buffers=batch_distributed_buffers,
    check=check,
    sequential_proposer=sequential_proposer,
)
