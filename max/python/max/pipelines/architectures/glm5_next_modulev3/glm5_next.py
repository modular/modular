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

from max import tree
from max.driver import CPU
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Module, as_subgraph
from max.experimental.nn.common_layers.embedding import VocabParallelEmbedding
from max.experimental.nn.common_layers.kv_cache import PagedCacheValues
from max.experimental.nn.common_layers.linear import ColumnParallelLinear
from max.experimental.nn.common_layers.mesh_axis import TP
from max.experimental.nn.common_layers.rotary_embedding import RotaryEmbedding
from max.experimental.nn.sequential import ModuleList
from max.experimental.sharding import DeviceMapping, NamedMapping
from max.experimental.tensor import Tensor
from max.graph import BufferValue, TensorValue
from max.nn.comm.ep import EPBatchManager, EPCommBuffers
from max.nn.kv_cache import (
    KVCacheInputsPerDevice,
    KVCacheParamInterface,
    RecurrentStateInputsPerDevice,
)
from max.pipelines.architectures.glm5_next.model_config import SPARSE_ATTENTION
from max.pipelines.architectures.glm5_next.state_cache import (
    CONV_LEAF_ID,
    INDEXER_CACHE_KEY,
    MLA_CACHE_KEY,
    RECURRENT_LEAF_ID,
    STATE_CACHE_KEY,
    STATE_LEAF_ORDER,
    TAIL_RING_LEAF_ID,
)
from max.pipelines.lib import ModuleV3Outputs

from ..deepseekV3_2_modulev3.deepseekV3_2 import project_to_logits
from ..deepseekV3_2_modulev3.layers.rms_norm import MultiplyBeforeCastRMSNorm
from ..deepseekV3_modulev3.deepseekV3 import gather_last_tokens
from .layers.decoder import Glm5NextDecoderLayer
from .layers.indexer import KPoolTail
from .layers.kda import KdaStateAccess, KimiDeltaAttention
from .layers.mlp import build_mlp
from .layers.sparse_mla import Glm5NextSparseMLA
from .model_config import Glm5NextConfig

__all__ = ["Glm5Next"]


def _state_leaf(
    per_device: list[RecurrentStateInputsPerDevice[TensorValue, BufferValue]],
    leaf_id: str,
    mapping: DeviceMapping,
    *,
    head_sharded: bool,
) -> tuple[Tensor, Tensor]:
    """Builds one state leaf's pool and rows as distributed tensors.

    A KDA pool holds the state of its own device's heads; the k-pool tail ring
    is the same on every device, since the indexer is replicated. The rows of
    a request are the same on every device.
    """
    index = STATE_LEAF_ORDER.index(leaf_id)
    pools = [device.leaves[index].pool for device in per_device]
    pool_mapping: DeviceMapping = mapping
    if head_sharded:
        spec = (None, TP) + (None,) * (pools[0].rank - 2)
        pool_mapping = NamedMapping(mapping.mesh, spec)
    pool = Tensor.from_shard_values(pools, pool_mapping)
    rows = Tensor.from_shard_values(
        [device.leaves[index].live_row_ids for device in per_device], mapping
    )
    return pool, rows


class Glm5NextTextModel(Module[..., Tensor]):
    """Embedding, the hybrid layer stack and the head."""

    def __init__(
        self,
        config: Glm5NextConfig,
        ep_batch_manager: EPBatchManager | None,
    ) -> None:
        if not config.layer_types:
            raise ValueError("Glm5NextConfig.layer_types is empty.")
        self.config = config
        self.mesh = config.mesh
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, dim=config.hidden_size
        )
        self.norm = MultiplyBeforeCastRMSNorm(
            dim=config.hidden_size, eps=config.rms_norm_eps
        )
        self.lm_head = ColumnParallelLinear(
            in_dim=config.hidden_size, out_dim=config.vocab_size, bias=False
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
                    mlp=build_mlp(config, layer_idx, ep_batch_manager),
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
        comm_buffers: EPCommBuffers | None,
    ) -> Tensor:
        config = self.config
        h = self.embed_tokens(tokens)
        mapping = tokens.mapping
        conv_pool, conv_rows = _state_leaf(
            state, CONV_LEAF_ID, mapping, head_sharded=True
        )
        recurrent_pool, recurrent_rows = _state_leaf(
            state, RECURRENT_LEAF_ID, mapping, head_sharded=True
        )
        tail_pool, tail_rows = _state_leaf(
            state, TAIL_RING_LEAF_ID, mapping, head_sharded=False
        )
        freqs_cis = F.cast(self.rope.freqs_cis, h.dtype).to(mapping)

        streams = F.broadcast_to(
            h.unsqueeze(1), [h.shape[0], config.hc_mult, h.shape[1]]
        )
        kda_position = 0
        sparse_position = 0
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
                    ),
                    input_row_offsets,
                )
            # Layers of one kind share a subgraph: the schedule fixes both the
            # attention family and dense-vs-MoE, and each call resolves its
            # own weights and passes its own cache and state positions.
            moe = (
                "moe" if layer_idx >= config.first_k_dense_replace else "dense"
            )
            call = as_subgraph(layer, name=f"glm5_next_{layer_type}_{moe}")
            streams = call(streams, attn_args, comm_buffers)

        h = F.mean(streams, axis=1).squeeze(1)
        return project_to_logits(
            gather_last_tokens(h, input_row_offsets),
            self.norm,
            self.lm_head,
            self.mesh,
            config.vocab_size,
            None,
        )


class Glm5Next(Module[..., ModuleV3Outputs]):
    """GLM-5.3-Flash with the flattened cache inputs unpacked."""

    def __init__(
        self,
        config: Glm5NextConfig,
        kv_params: KVCacheParamInterface,
        ep_batch_manager: EPBatchManager | None = None,
    ) -> None:
        super().__init__()
        self.language_model = Glm5NextTextModel(config, ep_batch_manager)
        self.config = config
        self.kv_params = kv_params
        self.ep_batch_manager = ep_batch_manager

    def forward(
        self,
        tokens: Tensor,
        return_n_logits: Tensor,
        input_row_offsets: Tensor,
        *variadic_args: Tensor,
    ) -> ModuleV3Outputs:
        del return_n_logits
        mesh = self.config.mesh
        assert mesh is not None
        if mesh.num_devices > 1:
            tokens = F.distributed_broadcast(tokens, mesh)
            input_row_offsets = F.distributed_broadcast(input_row_offsets, mesh)
        else:
            tokens = tokens.to(mesh)
            input_row_offsets = input_row_offsets.to(mesh)

        # The batch context length comes first; this graph does not read it.
        kv_inputs = iter(x._graph_value for x in variadic_args[1:])
        kv_tree = self.kv_params.unflatten_kv_inputs(kv_inputs)
        assert isinstance(kv_tree, dict)
        if STATE_CACHE_KEY not in kv_tree:
            raise ValueError(
                "The GLM-5.3-Flash graph needs KDA layers; the cache declared "
                "no recurrent state child."
            )

        def collection(name: str) -> PagedCacheValues:
            leaves = tree.leaves(kv_tree[name], leaf=KVCacheInputsPerDevice)
            return PagedCacheValues.from_upstream(leaves, tokens.mapping)

        state = tree.leaves(
            kv_tree[STATE_CACHE_KEY], leaf=RecurrentStateInputsPerDevice
        )
        comm_buffers: EPCommBuffers | None = None
        if self.ep_batch_manager is not None:
            comm_buffers = self.ep_batch_manager.comm_buffers(list(kv_inputs))
        logits = self.language_model(
            tokens,
            input_row_offsets,
            collection(MLA_CACHE_KEY),
            collection(INDEXER_CACHE_KEY),
            state,
            comm_buffers,
        )
        return ModuleV3Outputs(next_token_logits=logits)
