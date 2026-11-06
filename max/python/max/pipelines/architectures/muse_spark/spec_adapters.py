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
"""Block spec-decode adapters for a Muse Spark target and its DSpark draft.

The DSpark draft (``dspark_muse_spark``) embeds its block through the
target's table, so its proposer lives here, beside the target it reads. The
draft builds on this package's layers and so is not imported here.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from max import functional as F
from max.dtype import DType
from max.graph import BufferType, TensorType, TensorValue, ops
from max.nn.common_layers.kv_cache import PagedCacheValues
from max.nn.outputs import ReturnHiddenStates
from max.pipelines.lib.vlm_utils import F_merge_multimodal_embeddings
from max.pipelines.speculative.block_driver import (
    Accepted,
    BlockBatch,
    DraftSampler,
    block_kv_with_dispatch,
)
from max.pipelines.speculative.spec_adapters.target_adapter import (
    TargetAdapter,
)
from max.pipelines.speculative.spec_target import Verified
from max.tensor import Tensor
from max.tensor.sharding import DeviceMapping, DeviceMesh

from .layers.residual_gate import MuseSparkResidualBoundary
from .model_config import MuseSparkConfig
from .modules import MuseSparkDecoder
from .muse_spark import MuseSparkTextModel, replicate

if TYPE_CHECKING:
    from max.pipelines.speculative.block_driver import BlockProposer
    from max.pipelines.speculative.config import SpeculativeConfig
    from max.pipelines.speculative.driver import SequentialBatch
    from max.pipelines.speculative.spec_adapters import VerifyFn

FULL_KV = "full_attention"
SLIDING_KV = "sliding_attention"


def _from_shards(values: Sequence[TensorValue], mesh: DeviceMesh) -> Tensor:
    """One replicated tensor from each device's copy."""
    return Tensor.from_shard_values(
        list(values), DeviceMapping.replicated(mesh)
    )


def _shards(value: Tensor) -> list[TensorValue]:
    return [TensorValue(shard) for shard in value.local_shards]


def _kv(values: Sequence[Any], mesh: DeviceMesh) -> PagedCacheValues:
    return PagedCacheValues.from_upstream(
        list(values), DeviceMapping.replicated(mesh)
    )


def _verify_and_tap(
    model: MuseSparkTextModel,
    target_layer_ids: Sequence[int],
    tokens: Tensor,
    kv_by_type: dict[str, PagedCacheValues],
    return_n_logits: Tensor,
    input_row_offsets: Tensor,
    image_embeddings: Tensor,
    image_token_indices: Tensor,
) -> tuple[Tensor, Tensor]:
    """Returns each request's last logits and every row's taps.

    The logits are the last ``return_n_logits`` rows' of each request; the
    taps are the outputs of ``target_layer_ids``, concatenated along the
    hidden axis.
    """
    tokens = replicate(tokens, model.mesh)
    input_row_offsets = replicate(input_row_offsets, model.mesh)
    h = model.embed_norm(model.embed_tokens(tokens))
    # Images ride only context steps, whose merged rows are the prompt
    # rows the indices address; a step without images scatters nothing.
    h = F_merge_multimodal_embeddings(
        h, image_embeddings.cast(h.dtype), image_token_indices
    )
    model.prepare_freq_cis(model.mesh)
    taps = []
    for idx, layer in enumerate(model.layers):
        h = layer(
            h,
            kv_by_type[model._layer_kv_key[idx]],
            input_row_offsets=input_row_offsets,
        )
        if idx in target_layer_ids:
            if isinstance(h, MuseSparkResidualBoundary):
                h = h.join()
            taps.append(h)
    distance = F.range(
        return_n_logits[0],
        0,
        -1,
        out_dim="return_n_logits_range",
        device=model.mesh,
        dtype=DType.int64,
    )
    rows = (input_row_offsets[1:].unsqueeze(-1) - distance).reshape((-1,))
    logits = model._compute_logits(model.norm(F.gather(h, rows, axis=0)))
    return logits, F.concat(taps, axis=-1)


class MuseSparkBlockTarget:
    """Verifies the merged anchor and drafts on both target caches."""

    def __init__(
        self,
        target: MuseSparkTextModel,
        target_layer_ids: Sequence[int],
    ) -> None:
        self.target = target
        self.target_layer_ids = tuple(target_layer_ids)
        self.mesh = target.mesh

    def __call__(
        self, batch: SequentialBatch | BlockBatch
    ) -> Verified[list[TensorValue]]:
        """Runs the target over ``batch``'s merged rows: logits and taps."""
        assert isinstance(batch, BlockBatch)
        if self.target.ep_batch_manager is not None:
            assert batch.ep_inputs is not None
            self.target.bind_ep_buffers(batch.ep_inputs)
        logits, taps = _verify_and_tap(
            self.target,
            self.target_layer_ids,
            Tensor.from_graph_value(batch.merged_tokens),
            {
                FULL_KV: _kv(batch.kv_collections, self.mesh),
                SLIDING_KV: _kv(batch.passthrough_kv[SLIDING_KV], self.mesh),
            },
            Tensor.from_graph_value(batch.return_n_logits),
            Tensor.from_graph_value(batch.merged_offsets),
            _from_shards(batch.vision_embeddings, self.mesh),
            _from_shards(batch.vision_scatter_indices, self.mesh),
        )
        return Verified(logits=_shards(logits)[0], hidden=_shards(taps))


class MuseSparkDSparkProposer:
    """The drafter's block: the target's embedding, the drafter's head.

    Every slot drafts, the anchor's included (``sample_from_anchor``).
    """

    samples_from_anchor = True
    supports_zero_draft_rows = False

    def __init__(
        self,
        draft: Any,
        target: MuseSparkTextModel,
        *,
        num_drafts: int,
    ) -> None:
        self.draft = draft
        self.target = target
        self.mesh = target.mesh
        self.block_size = draft.config.spec.block_size
        self.mask_token_id = draft.config.spec.mask_token_id
        self.num_drafts = num_drafts

    def materialize(
        self,
        batch: BlockBatch,
        target_hidden: list[TensorValue],
        ctx_kv: list[Any],
    ) -> None:
        """Writes the drafter's context KV from the target's taps."""
        # Every merged row is written, rejected drafts' included: the cache
        # advances by the committed rows only, so the next step's context
        # overwrites the rest.
        self.draft.prepare_freq_cis(self.mesh)
        self.draft.materialize(
            _from_shards(target_hidden, self.mesh),
            _kv(ctx_kv, self.mesh),
            _from_shards(batch.merged_offsets_per_dev, self.mesh),
        )

    def embed_block(
        self, batch: BlockBatch, block_ids: TensorValue
    ) -> list[TensorValue]:
        """Embeds ``block_ids`` with the target's embedding, per device."""
        ids = replicate(Tensor.from_graph_value(block_ids), self.mesh)
        return _shards(self.target.embed_tokens(ids))

    def forward_block(
        self,
        batch: BlockBatch,
        embeds: list[TensorValue],
        offsets: list[TensorValue],
        block_kv: list[Any],
    ) -> list[TensorValue]:
        """Runs the drafter over the embedded block, per device."""
        hidden = self.draft(
            _from_shards(embeds, self.mesh),
            _kv(block_kv_with_dispatch(block_kv, self.block_size), self.mesh),
            _from_shards(offsets, self.mesh),
        )
        return _shards(hidden)

    def head(
        self,
        batch: BlockBatch,
        block_hs: list[TensorValue],
        accepted: Accepted,
        sampler: DraftSampler,
    ) -> TensorValue:
        """Samples the block's drafts from the drafter's logits."""
        hidden = _from_shards(block_hs, self.mesh)
        base = self.draft.lm_head(hidden)
        base = base.reshape([-1, self.block_size, int(base.shape[-1])])
        # Sequential: each slot's bias reads the token drawn before it, the
        # anchor's for the first.
        previous = accepted.next_tokens
        sampled = []
        for step in range(self.num_drafts):
            logits = self.draft.slot_logits(
                base[:, step, :],
                self.draft.markov_embed(
                    replicate(Tensor.from_graph_value(previous), self.mesh)
                ),
            )
            previous = sampler.sample_next(_shards(logits)[0], step)
            sampled.append(ops.unsqueeze(previous, axis=1))
        return ops.concat(sampled, axis=1)


def verify(
    target: Any,
    target_config: Any,
    hidden_states: ReturnHiddenStates,
    draft_config: Any,
) -> VerifyFn:
    """Returns the verify on both target caches, tapping the draft's layers."""
    del hidden_states, draft_config
    assert isinstance(target, MuseSparkDecoder)
    assert isinstance(target_config, MuseSparkConfig)
    return MuseSparkBlockTarget(
        target.model.language_model,
        target_config.target_layer_ids,
    )


def takes_vision_outputs(target_config: Any) -> bool:
    """Returns ``True``: the graph takes the image rows Mach's tower encodes.

    MAX registers the pairing for text only, so it refuses image requests and
    binds zero rows; the serving engine runs the encoder and passes its rows.
    """
    del target_config
    return True


def ep_input_types(target: Any) -> Sequence[TensorType | BufferType]:
    """Returns the EP buffers, which the graph takes past its tail."""
    assert isinstance(target, MuseSparkDecoder)
    manager = target.model.ep_batch_manager
    return () if manager is None else tuple(manager.input_types())


def block_proposer(
    verify: VerifyFn,
    draft: Any,
    draft_config: Any,
    speculative: SpeculativeConfig,
) -> BlockProposer[list[TensorValue], list[TensorValue]]:
    """Returns the DSpark proposer, which embeds through the target."""
    del draft_config
    assert isinstance(verify, MuseSparkBlockTarget)
    return MuseSparkDSparkProposer(
        draft, verify.target, num_drafts=speculative.draft_width
    )


ADAPTER = TargetAdapter(
    primary_kv=FULL_KV,
    verify=verify,
    ep_input_types=ep_input_types,
    takes_vision_outputs=takes_vision_outputs,
    # Mach's Muse Spark DSpark graph takes no ``in_thinking_phase``.
    relaxed_thinking_acceptance=False,
    block_proposer=block_proposer,
)
