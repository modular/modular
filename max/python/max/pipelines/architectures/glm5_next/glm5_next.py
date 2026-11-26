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
"""The GLM-5.3-Flash decoder on ModuleV3.

The period-4 hybrid schedule -- three KDA layers, then one sparse-MLA layer --
over a residual widened to ``hc_mult`` streams. The streams are created once
after the embedding and collapsed by an unweighted mean before the final norm;
GLM's collapse is not learned, unlike DeepSeek-V4's.
"""

from __future__ import annotations

from dataclasses import dataclass

from max import functional as F
from max import tree
from max.driver import CPU
from max.dtype import DType
from max.graph import BufferValue, TensorValue
from max.nn import Module, as_subgraph
from max.nn.common_layers.embedding import VocabParallelEmbedding
from max.nn.common_layers.kv_cache import PagedCacheValues, kv_input_values
from max.nn.common_layers.linear import ColumnParallelLinear
from max.nn.common_layers.mesh_axis import DP, TP
from max.nn.common_layers.rotary_embedding import RotaryEmbedding
from max.nn.ep import EPBuffers
from max.nn.ep_config import EPConfig
from max.nn.kv_cache import (
    KVCacheInputsPerDevice,
    KVCacheParamInterface,
    RecurrentStateInputsPerDevice,
)
from max.nn.sequential import ModuleList
from max.pipelines.lib import ModuleV3Outputs
from max.pipelines.lib.role_contracts import VisionInputs
from max.pipelines.lib.utils import split_replicated_batch
from max.pipelines.lib.vlm_utils import F_merge_multimodal_embeddings
from max.tensor import Tensor
from max.tensor.sharding import (
    DeviceMapping,
    DeviceMesh,
    NamedMapping,
    Replicated,
    Sharded,
    Unknown,
)

from .hybrid_layout import SPARSE_ATTENTION
from .layers.decoder import Glm5NextDecoderLayer, Pending, write_back
from .layers.indexer import KPoolTail
from .layers.kda import KdaStateAccess, KimiDeltaAttention
from .layers.mlp import build_mlp
from .layers.norm import MultiplyBeforeCastRMSNorm
from .layers.sparse_mla import Glm5NextSparseMLA
from .logits import gather_last_tokens, project_to_logits
from .model_config import Glm5NextConfig
from .state_cache import (
    CONV_LEAF_ID,
    INDEXER_CACHE_KEY,
    MLA_CACHE_KEY,
    RECURRENT_LEAF_ID,
    SPEC_STATE_LEAF_ORDER,
    STATE_CACHE_KEY,
    TAIL_RING_LEAF_ID,
)

__all__ = [
    "Glm5Next",
    "Glm5NextTextModel",
    "is_data_parallel",
    "replica_mapping",
    "replicate",
    "state_leaf",
]


def is_data_parallel(mesh: DeviceMesh) -> bool:
    """Returns whether ``mesh`` runs one attention replica per device."""
    return DP in mesh.axis_names


def replica_mapping(mesh: DeviceMesh) -> DeviceMapping:
    """Returns the mapping of per-request device state on ``mesh``.

    Data-parallel replicas serve different requests, so their caches, state
    pools and row tables vary by device; tensor-parallel devices all hold
    the same rows.
    """
    return DeviceMapping(
        mesh,
        tuple(
            Unknown() if name == DP else Replicated()
            for name in mesh.axis_names
        ),
    )


def replicate(value: Tensor, mesh: DeviceMesh) -> Tensor:
    """Places a single-device tensor on every device of ``mesh``.

    The broadcast collective needs signal buffers, which only a multi-device
    graph has. Onto one device, replicating is a placement.
    """
    if mesh.num_devices > 1:
        return F.distributed_broadcast(value, mesh)
    return value.to(mesh)


def state_leaf(
    per_device: list[RecurrentStateInputsPerDevice[TensorValue, BufferValue]],
    leaf_id: str,
    mapping: DeviceMapping,
    *,
    head_sharded: bool,
) -> tuple[Tensor, Tensor]:
    """Builds one state leaf's pool and rows as distributed tensors.

    A KDA pool holds the state of its own device's heads; the k-pool tail ring
    is the same on every device, since the indexer is replicated. The rows of
    a request are the same on every device. A speculative cache appends the
    verify ring, head-split like the KDA state, after the live leaves.

    Under data parallelism each device holds every head of its own replica's
    requests, so ``mapping`` is the replicas' (:func:`replica_mapping`) and
    no pool is head-split.
    """
    index = SPEC_STATE_LEAF_ORDER.index(leaf_id)
    pools = [device.leaves[index].pool for device in per_device]
    pool_mapping: DeviceMapping = mapping
    if head_sharded and not is_data_parallel(mapping.mesh):
        spec = (None, TP) + (None,) * (pools[0].rank - 2)
        pool_mapping = NamedMapping(mapping.mesh, spec)
    pool = Tensor.from_shard_values(pools, pool_mapping)
    rows = Tensor.from_shard_values(
        [device.leaves[index].live_row_ids for device in per_device], mapping
    )
    return pool, rows


@dataclass(frozen=True)
class DataParallelBatch:
    """The inputs that split a batch into its data-parallel replicas."""

    host_input_row_offsets: Tensor
    """The batch's row offsets, int64, on the host."""
    splits: Tensor
    """The first request of each replica, then the batch size, on the host."""


def _check_data_parallel_experts(
    config: Glm5NextConfig, ep_config: EPConfig | None
) -> None:
    """Raises unless the MoE layers can run on per-replica rows.

    Replicas hold different rows, so the tensor-parallel MoE's all-reduce
    and the all-reduce expert exchange, which both sum every device's
    partials of the same rows, would add replicas' rows together. The P2P
    exchange dispatches each device's own rows.
    """
    has_moe = config.num_hidden_layers > config.first_k_dense_replace
    if not has_moe:
        return
    if ep_config is None:
        raise ValueError(
            "GLM-5.3-Flash under data parallelism runs its experts "
            "expert-parallel: set --ep-size to the device count."
        )
    if ep_config.use_allreduce:
        raise ValueError(
            "GLM-5.3-Flash under data parallelism needs the P2P expert "
            "exchange: drop --ep-use-allreduce."
        )


class Glm5NextTextModel(Module[..., Tensor]):
    """Embedding, the hybrid layer stack and the head."""

    def __init__(
        self,
        config: Glm5NextConfig,
        ep_config: EPConfig | None,
    ) -> None:
        if not config.layer_types:
            raise ValueError("Glm5NextConfig.layer_types is empty.")
        self.config = config
        self.mesh = config.mesh
        self.ep_config = ep_config
        self.data_parallel = config.data_parallel_degree > 1
        if self.data_parallel:
            _check_data_parallel_experts(config, ep_config)
        # Under data parallelism the attention weights hold whole heads on
        # every device (their ``tp`` specs resolve to replicated on a ``dp``
        # mesh); the vocabulary shards over the ``dp`` axis instead, as
        # DeepSeek-V3.2's does, so the embedding and head are not copied
        # eight times.
        vocab_axis = DP if self.data_parallel else None
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, dim=config.hidden_size, tp_axis=vocab_axis
        )
        self.norm = MultiplyBeforeCastRMSNorm(
            dim=config.hidden_size, eps=config.rms_norm_eps
        )
        self.lm_head = ColumnParallelLinear(
            in_dim=config.hidden_size,
            out_dim=config.vocab_size,
            bias=False,
            tp_axis=vocab_axis,
        )
        # The padded MLA kernels read the rotary width off this table's shape.
        # Its values are never used -- every column it rotates is zero -- so
        # one table serves every sparse layer.
        rope_width = config.mla_latent_pad_to - config.kv_lora_rank
        self.rope = RotaryEmbedding(
            dim=rope_width,
            n_heads=config.num_attention_heads,
            theta=10000.0,
            max_seq_len=config.max_seq_len,
            device=config.devices[0].to_device(),
            head_dim=rope_width,
            interleaved=False,
        )

        quant_scheme = config.quant_scheme
        if quant_scheme is None:
            raise ValueError(
                "GLM-5.3-Flash needs a resolved quant scheme: its precision "
                "map is read off the checkpoint's weight scales."
            )
        layers = []
        for layer_idx, layer_type in enumerate(config.layer_types):
            self_attn: Module[..., Tensor]
            if layer_type == SPARSE_ATTENTION:
                self_attn = Glm5NextSparseMLA(
                    config, layer_idx, quant_scheme.attn_config(layer_idx)
                )
            else:
                self_attn = KimiDeltaAttention(config)
            layers.append(
                Glm5NextDecoderLayer(
                    config,
                    self_attn=self_attn,
                    mlp=build_mlp(config, layer_idx, ep_config),
                )
            )
        self.layers = ModuleList[Glm5NextDecoderLayer](layers)

    def forward(
        self,
        tokens: Tensor,
        input_row_offsets: Tensor,
        mla: PagedCacheValues,
        indexer: PagedCacheValues,
        state: list[RecurrentStateInputsPerDevice[TensorValue, BufferValue]],
        ep: EPBuffers | None,
        vision: VisionInputs[Tensor] | None = None,
        data_parallel: DataParallelBatch | None = None,
    ) -> Tensor:
        """Returns the logits at each request's last token.

        Under data parallelism the logits come back in batch order: the
        replicas' rows in replica order, as the batch lists them.
        """
        h, input_row_offsets = self.embed(
            tokens, input_row_offsets, vision, data_parallel
        )
        h = self.run_layers(h, input_row_offsets, mla, indexer, state, ep)
        last = gather_last_tokens(h, input_row_offsets)
        if self.data_parallel:
            # The vocab-sharded head needs every replica's rows on every
            # device.
            last = F.allgather(last, mesh_axis=DP)
        return project_to_logits(
            last,
            self.norm,
            self.lm_head,
            self.mesh,
            self.config.vocab_size,
            None,
        )

    def embed(
        self,
        tokens: Tensor,
        input_row_offsets: Tensor,
        vision: VisionInputs[Tensor] | None,
        data_parallel: DataParallelBatch | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Returns the embedded rows each device runs, and their offsets.

        Image rows replace the embeddings at their placeholder positions,
        which is the whole vision-to-language interface: the decoder is NoPE,
        so no position ids change. The merge runs on the whole batch, whose
        scatter indices are batch-global, before a data-parallel split cuts
        out each replica's rows and rebases their offsets to zero.
        """
        h = self.embed_tokens(tokens)
        if vision is not None:
            h = F_merge_multimodal_embeddings(
                h, vision.embeddings, vision.scatter_indices
            )
        if not self.data_parallel:
            return h, input_row_offsets
        if data_parallel is None:
            raise ValueError(
                "A data-parallel GLM-5.3-Flash graph needs the batch's "
                "data_parallel_splits and host row offsets."
            )
        mesh = h.mesh
        return split_replicated_batch(
            h,
            input_row_offsets,
            data_parallel.host_input_row_offsets,
            data_parallel.splits,
            DeviceMapping(
                mesh,
                tuple(
                    Sharded(0) if name == DP else Replicated()
                    for name in mesh.axis_names
                ),
            ),
        )

    def hidden_states(
        self,
        tokens: Tensor,
        input_row_offsets: Tensor,
        mla: PagedCacheValues,
        indexer: PagedCacheValues,
        state: list[RecurrentStateInputsPerDevice[TensorValue, BufferValue]],
        ep: EPBuffers | None,
        *,
        vision: VisionInputs[Tensor] | None = None,
        subgraphs: bool = True,
    ) -> Tensor:
        """Returns every token's collapsed residual, before the final norm.

        A tensor-parallel batch only: the speculative verify calls it, and
        it does not run data-parallel. See :meth:`embed` for the vision
        merge.

        Args:
            tokens: The batch's token ids, ragged by ``input_row_offsets``.
            input_row_offsets: Offsets of each request's rows.
            mla: The sparse layers' latent cache.
            indexer: The sparse layers' indexer key cache.
            state: Each device's recurrent-state leaves.
            ep: The exchange's buffers, or ``None`` without expert
                parallelism.
            vision: The batch's image rows and their token positions, when
                the checkpoint ships a vision tower.
            subgraphs: Whether layers of one kind share a subgraph. A
                speculative verify turns it off: its rollback reads what each
                layer captures, which cannot leave a subgraph.
        """
        h, input_row_offsets = self.embed(tokens, input_row_offsets, vision)
        return self.run_layers(
            h,
            input_row_offsets,
            mla,
            indexer,
            state,
            ep,
            subgraphs=subgraphs,
        )

    def run_layers(
        self,
        h: Tensor,
        input_row_offsets: Tensor,
        mla: PagedCacheValues,
        indexer: PagedCacheValues,
        state: list[RecurrentStateInputsPerDevice[TensorValue, BufferValue]],
        ep: EPBuffers | None,
        *,
        subgraphs: bool = True,
    ) -> Tensor:
        """Runs the layer stack over the embedded rows ``h``.

        Returns every row's collapsed residual, before the final norm.
        """
        config = self.config
        assert self.mesh is not None
        mapping = (
            replica_mapping(self.mesh)
            if self.data_parallel
            else DeviceMapping.replicated(self.mesh)
        )
        conv_pool, conv_rows = state_leaf(
            state, CONV_LEAF_ID, mapping, head_sharded=True
        )
        recurrent_pool, recurrent_rows = state_leaf(
            state, RECURRENT_LEAF_ID, mapping, head_sharded=True
        )
        if not subgraphs:
            # The recurrence reads int32 rows. Cast once here, every layer's
            # row is a static slice of the table, which costs no kernel; a
            # per-layer cast costs one each.
            recurrent_rows = recurrent_rows.cast(DType.int32)
        tail_pool, tail_rows = state_leaf(
            state, TAIL_RING_LEAF_ID, mapping, head_sharded=False
        )
        freqs_cis = self.rope.freqs_cis.cast(h.dtype).to(
            DeviceMapping.replicated(self.mesh)
        )

        streams = h.unsqueeze(1).broadcast_to(
            [h.shape[0], config.hc_mult, h.shape[1]]
        )
        kda_position = 0
        sparse_position = 0
        pending: Pending | None = None
        for layer_idx, (layer_type, layer) in enumerate(
            zip(config.layer_types, self.layers, strict=True)
        ):
            attn_args: tuple[object, ...]
            if layer_type == SPARSE_ATTENTION:
                # The cache index, not the model index: only the sparse
                # layers hold a KV cache, so model layer 3 is cache layer 0.
                position = sparse_position
                sparse_position += 1
                attn_args = (
                    mla,
                    indexer,
                    F.constant(position, DType.uint32, device=CPU()),
                    input_row_offsets,
                    freqs_cis,
                    KPoolTail(
                        pool=tail_pool,
                        rows=tail_rows,
                        layer=F.constant(position, DType.int64, device=CPU()),
                    ),
                )
            else:
                position = kda_position
                kda_position += 1
                attn_args = (
                    KdaStateAccess(
                        conv_pool=conv_pool,
                        conv_rows=conv_rows,
                        recurrent_pool=recurrent_pool,
                        recurrent_rows=recurrent_rows,
                        layer=F.constant(position, DType.int64, device=CPU()),
                        position=None if subgraphs else position,
                    ),
                    input_row_offsets,
                )
            # Layers of one kind share a subgraph: the schedule fixes both the
            # attention family and dense-vs-MoE, and each call resolves its
            # own weights and passes its own cache and state positions.
            moe = (
                "moe" if layer_idx >= config.first_k_dense_replace else "dense"
            )
            # The first layer has no pending write back, so it gets a
            # subgraph of its own.
            first = "_first" if pending is None else ""
            call = (
                as_subgraph(layer, name=f"glm5_next_{layer_type}_{moe}{first}")
                if subgraphs
                else layer
            )
            streams, pending = call(streams, attn_args, ep, pending)

        assert pending is not None
        streams = write_back(streams, *pending)
        return F.mean(streams, axis=1).squeeze(1)


class Glm5Next(Module[..., ModuleV3Outputs]):
    """GLM-5.3-Flash with the flattened cache inputs unpacked."""

    def __init__(
        self,
        config: Glm5NextConfig,
        kv_params: KVCacheParamInterface,
        ep_config: EPConfig | None = None,
    ) -> None:
        super().__init__()
        self.language_model = Glm5NextTextModel(config, ep_config)
        self.config = config
        self.kv_params = kv_params
        self.ep_config = ep_config

    def forward(
        self,
        tokens: Tensor,
        return_n_logits: Tensor,
        input_row_offsets: Tensor,
        *variadic_args: Tensor,
        vision: VisionInputs[Tensor] | None = None,
        ep: EPBuffers | None = None,
        host_input_row_offsets: Tensor | None = None,
        data_parallel_splits: Tensor | None = None,
    ) -> ModuleV3Outputs:
        """Runs the language graph over the batch's tokens and images.

        Under data parallelism it also takes the batch's host row offsets
        and the first request of each replica (``data_parallel_splits``).
        """
        del return_n_logits
        mesh = self.config.mesh
        assert mesh is not None
        tokens = replicate(tokens, mesh)
        input_row_offsets = replicate(input_row_offsets, mesh)
        data_parallel: DataParallelBatch | None = None
        if self.config.data_parallel_degree > 1:
            assert host_input_row_offsets is not None
            assert data_parallel_splits is not None
            data_parallel = DataParallelBatch(
                host_input_row_offsets=host_input_row_offsets.cast(DType.int64),
                splits=data_parallel_splits,
            )

        # The batch context length comes first; this graph does not read it.
        kv_inputs = iter(kv_input_values(variadic_args[1:]))
        kv_tree = self.kv_params.unflatten_kv_inputs(kv_inputs)
        assert isinstance(kv_tree, dict)
        if STATE_CACHE_KEY not in kv_tree:
            raise ValueError(
                "The GLM-5.3-Flash graph needs KDA layers; the cache declared "
                "no recurrent state child."
            )

        cache_mapping = (
            replica_mapping(mesh)
            if data_parallel is not None
            else tokens.mapping
        )

        def collection(name: str) -> PagedCacheValues:
            leaves = tree.leaves(kv_tree[name], leaf=KVCacheInputsPerDevice)
            return PagedCacheValues.from_upstream(leaves, cache_mapping)

        state = tree.leaves(
            kv_tree[STATE_CACHE_KEY], leaf=RecurrentStateInputsPerDevice
        )
        logits = self.language_model(
            tokens,
            input_row_offsets,
            collection(MLA_CACHE_KEY),
            collection(INDEXER_CACHE_KEY),
            state,
            ep,
            vision,
            data_parallel,
        )
        return ModuleV3Outputs(next_token_logits=logits)
