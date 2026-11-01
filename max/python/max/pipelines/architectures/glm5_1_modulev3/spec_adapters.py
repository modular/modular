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
"""The ModuleV3 GLM-5.x decoder's call, and its NextN layer's proposer.

Both run on ``Tensor`` values: the proposer is a
:class:`~max.pipelines.speculative.driver.TensorAdapter`, and the verify
converts its batch at its own boundary. The driver hands them ``Tensor``
values, one per device where it holds a list, and they build the distributed
tensors the modules take from those shards.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from max import functional as F
from max.graph import BufferType, DimLike, TensorType
from max.nn.common_layers.kv_cache import PagedCacheValues
from max.nn.common_layers.multi_latent_attention import (
    MLAPrefillMetadata,
)
from max.nn.legacy.comm.ep import EPCommBuffers
from max.nn.legacy.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.speculative._tensor_compat import (
    as_graph_values,
    as_tensors,
)
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
from max.pipelines.speculative.spec_target import Verified
from max.tensor import Tensor
from max.tensor.sharding import DeviceMapping, DeviceMesh

from ..deepseekV3_2.layers.sparse_mla import sparse_selection_width
from ..deepseekV3_2_modulev3.deepseekV3_2 import project_to_logits
from ..deepseekV3_2_modulev3.nextn import DeepseekV3_2NextN
from ..deepseekV3_modulev3.deepseekV3 import gather_last_tokens
from .glm5_1 import Glm5_1Decoder

if TYPE_CHECKING:
    from max.pipelines.speculative.driver import SequentialProposer
    from max.pipelines.speculative.spec_adapters import VerifyFn

__all__ = ["ADAPTER", "Glm5_1MTPProposer", "Glm5_1Target"]

_Batch = SequentialBatch[Tensor, Tensor, Tensor]

PRIMARY_KV = "mla"

_INDEXER_KV = "indexer"
"""The target's indexer cache, which the driver passes through."""

_DRAFT_INDEXER_KV = "draft_indexer"
"""The NextN layer's own indexer cache, which the driver passes through."""


def _replicate(tokens: Tensor, mesh: DeviceMesh) -> Tensor:
    if mesh.num_devices > 1:
        return F.distributed_broadcast(tokens, mesh)
    # The broadcast collective needs signal buffers, which only a
    # multi-device graph has. Onto one device, replicating is a placement.
    return tokens.to(mesh)


class Glm5_1Target:
    """The GLM-5.x decoder's verify, and its output layout.

    Verification reads VARIABLE logits and every token's normalized hidden
    state, and skips the last-token logits nothing reads.
    """

    takes_distributed_inputs = True

    def __init__(self, target: Glm5_1Decoder) -> None:
        self.target = target
        language_model = target.language_model
        language_model.return_logits = ReturnLogits.VARIABLE
        language_model.return_hidden_states = ReturnHiddenStates.ALL_NORMALIZED
        language_model.emit_last_token_logits = False

    def __call__(self, batch: SequentialBatch | BlockBatch) -> Verified[Any]:
        """Verifies the batch's drafts against the target."""
        # Only the NextN layer drafts for this target, a token a step.
        assert isinstance(batch, SequentialBatch)
        verified: Verified[Any] = as_graph_values(
            self._verify(as_tensors(batch))
        )
        return verified

    def _verify(self, batch: _Batch) -> Verified[list[Tensor], Tensor]:
        language_model = self.target.language_model
        mesh = language_model.mesh
        assert mesh is not None
        # Tensor-parallel devices all store the same latent cache.
        replicated = DeviceMapping.replicated(mesh)

        comm_buffers = None
        ep_batch_manager = self.target.ep_batch_manager
        if ep_batch_manager is not None:
            assert batch.ep_inputs is not None
            comm_buffers = ep_batch_manager.comm_buffers(batch.ep_inputs)

        logits, _, hidden = language_model(
            batch.merged_tokens,
            PagedCacheValues.from_upstream(batch.kv_collections, replicated),
            PagedCacheValues.from_upstream(
                batch.passthrough_kv[_INDEXER_KV], replicated
            ),
            batch.return_n_logits,
            # The target broadcasts the merged tokens and offsets itself.
            batch.merged_offsets,
            batch.batch_context_lengths[0],
            None,
            None,
            comm_buffers,
        )
        return Verified(logits=logits, hidden=list(hidden.local_shards))


class Glm5_1MTPProposer(TensorAdapter):
    """The sparse NextN layer, sharing the target's embedding and head.

    It embeds with the target's ``embed_tokens`` and projects with the
    target's ``lm_head`` after its own ``shared_head_norm``.
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

    def __init__(self, draft: DeepseekV3_2NextN, target: Glm5_1Decoder) -> None:
        self.draft = draft
        self.language_model = target.language_model
        mesh = self.language_model.mesh
        assert mesh is not None
        self.mesh = mesh
        self.hidden_dim: DimLike = draft.config.hidden_size
        self.reuse: ReuseSpec | None = ReuseSpec(
            dim=sparse_selection_width(
                draft.config.mla_kv_params, draft.config.index_topk
            ),
            split_prefix="mtp_topk",
        )

    def prefill(
        self, batch: _Batch, tokens: Tensor, target_hidden: list[Tensor]
    ) -> Proposed[Tensor]:
        """Runs the draft over the prompt's target hidden states."""
        h, topk, offsets = self._decode(batch, tokens, target_hidden, None)
        # Every row's hidden state and the logits of each request's last
        # ``return_n_logits`` rows, which the driver gathers at the accepted
        # position.
        rows, _ = gather_variable_rows(
            h, offsets, batch.return_n_logits, self.mesh
        )
        return Proposed(
            logits=self._project(rows),
            hidden=list(h.local_shards),
            reuse=list(topk.local_shards),
        )

    def step(
        self, batch: _Batch, draft_input: DraftStepInput[Tensor], index: int
    ) -> Proposed[Tensor]:
        """Runs one draft step from the previous step's tokens."""
        # ``hnorm`` expects the post-final-norm hidden state, which the target
        # hands over at step 0. ``shared_head_norm`` is row-wise, so
        # normalizing the one carried row here is equivalent and far cheaper.
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

    @property
    def _replicated(self) -> DeviceMapping:
        return DeviceMapping.replicated(self.mesh)

    def _project(self, h: Tensor) -> Tensor:
        config = self.draft.config
        return project_to_logits(
            h,
            self.draft.shared_head_norm,
            self.language_model.lm_head,
            config.mesh,
            config.vocab_size,
            config.unpadded_vocab_size,
        )

    def _decode(
        self,
        batch: _Batch,
        tokens: Tensor,
        hidden: list[Tensor] | Tensor,
        reuse: list[Tensor] | None,
        selection_lag: int = 0,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Runs the draft layer, returning its output, top-k and offsets."""
        replicated = self._replicated
        embedded = self.language_model.embed_tokens(
            _replicate(tokens, self.mesh)
        )
        if isinstance(hidden, list):
            hidden = Tensor.from_shard_values(hidden, replicated)
        offsets = Tensor.from_shard_values(
            batch.query_offsets_per_dev, replicated
        )
        kv_collection = PagedCacheValues.from_upstream(
            batch.draft_kv_collections, replicated
        )
        indexer_kv_collection = PagedCacheValues.from_upstream(
            batch.passthrough_kv[_DRAFT_INDEXER_KV], replicated
        )
        freqs_cis = F.cast(
            self.language_model.rope.freqs_cis, embedded.dtype
        ).to(self.mesh)

        attention = self.draft.decoder_layer.self_attn
        mla_prefill_metadata: MLAPrefillMetadata | None = None
        if attention.graph_mode in ("prefill", "auto"):
            mla_prefill_metadata = attention.create_mla_prefill_metadata(
                offsets, kv_collection
            )
            # Host-substitute the D2H buffer_length copy, as the target does.
            mla_prefill_metadata.buffer_lengths = batch.batch_context_lengths[0]

        comm_buffers: EPCommBuffers | None = None
        if self.draft.ep_batch_manager is not None:
            assert batch.ep_inputs is not None
            comm_buffers = self.draft.ep_batch_manager.comm_buffers(
                batch.ep_inputs
            )

        h, topk = self.draft(
            embedded,
            hidden,
            kv_collection,
            indexer_kv_collection,
            offsets,
            freqs_cis,
            mla_prefill_metadata,
            comm_buffers,
            Tensor.from_shard_values(reuse, replicated)
            if reuse is not None
            else None,
            reuse is not None,
            selection_lag=selection_lag,
        )
        return h, topk, offsets


def verify(
    target: Any,
    target_config: Any,
    hidden_states: ReturnHiddenStates,
    draft_config: Any,
) -> VerifyFn:
    """Returns the verify of a ModuleV3 GLM-5.x decoder.

    Raises:
        ValueError: If the target wasn't built to return the normalized
            hidden states the NextN layer reads, or spans several devices
            without expert parallelism.
        NotImplementedError: Under data parallelism.
    """
    del target_config, draft_config
    if hidden_states != ReturnHiddenStates.ALL_NORMALIZED:
        raise ValueError(
            "A ModuleV3 GLM-5.x target verifies only for a draft that reads"
            f" its normalized hidden states. Got {hidden_states.value}."
        )
    assert isinstance(target, Glm5_1Decoder)
    config = target.config
    if config.data_parallel_degree > 1:
        raise NotImplementedError(
            "MTP with a ModuleV3 GLM-5.x target requires"
            " data_parallel_degree=1."
        )
    if len(config.devices) > 1 and target.ep_batch_manager is None:
        raise ValueError(
            "MTP with a ModuleV3 GLM-5.x target on several devices requires"
            " expert parallelism over all of them."
        )
    return Glm5_1Target(target)


def ep_input_types(target: Any) -> Sequence[TensorType | BufferType]:
    """Returns the target's expert-parallel inputs, empty when EP is off."""
    assert isinstance(target, Glm5_1Decoder)
    if target.ep_batch_manager is None:
        return ()
    return target.ep_batch_manager.input_types()


def sequential_proposer(
    verify: VerifyFn,
    draft: Any,
    draft_config: Any,
    speculative: SpeculativeConfig,
) -> SequentialProposer[Any, Any, Any, Any]:
    """Returns the NextN layer's proposer, which reads the verified target."""
    del draft_config, speculative
    assert isinstance(verify, Glm5_1Target)
    assert isinstance(draft, DeepseekV3_2NextN)
    return Glm5_1MTPProposer(draft, verify.target)


def check(speculative: SpeculativeConfig, structured_output: bool) -> None:
    """Refuses every method but MTP."""
    del structured_output
    if not speculative.is_mtp():
        raise ValueError(
            "A ModuleV3 GLM-5.x target speculates only with its NextN layer,"
            f" got --speculative-method {speculative.speculative_method!r}."
        )


ADAPTER = TargetAdapter(
    primary_kv=PRIMARY_KV,
    verify=verify,
    ep_input_types=ep_input_types,
    check=check,
    sequential_proposer=sequential_proposer,
)
