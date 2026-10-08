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
"""Implements the DeepseekV3.2 model using the ModuleV3 API."""

from __future__ import annotations

import math
from collections.abc import Callable

from max import tree
from max.driver import CPU
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Module, as_subgraph
from max.experimental.nn.common_layers.embedding import VocabParallelEmbedding
from max.experimental.nn.common_layers.kv_cache import PagedCacheValues
from max.experimental.nn.common_layers.linear import ColumnParallelLinear
from max.experimental.nn.common_layers.mesh_axis import DP
from max.experimental.nn.common_layers.rotary_embedding import RotaryEmbedding
from max.experimental.nn.sequential import ModuleList
from max.experimental.sharding import (
    DeviceMapping,
    DeviceMesh,
    Replicated,
    Sharded,
    Unknown,
)
from max.experimental.tensor import Tensor
from max.graph import TensorValue
from max.nn.comm.ep import EPBatchManager, EPCommBuffers
from max.nn.kv_cache import KVCacheInputsPerDevice, KVCacheParamInterface
from max.nn.rotary_embedding import DeepseekYarnRopeScalingParams
from max.pipelines.lib import ModuleV3Outputs

from ..deepseekV2_modulev3.layers.rotary_embedding import (
    DeepseekYarnRotaryEmbedding,
)
from ..deepseekV3_modulev3.deepseekV3 import (
    gather_last_tokens,
    split_replicated_batch,
)
from ..deepseekV3_modulev3.layers.quant_moe import QuantizedMoE
from .layers.rms_norm import MultiplyBeforeCastRMSNorm
from .layers.transformer_block import DeepseekV3_2TransformerBlock
from .model_config import DeepseekV3_2Config


def mask_padded_tail(
    logits: Tensor, vocab_size: int, unpadded_vocab_size: int | None
) -> Tensor:
    """Sends the dummy/padding rows of the vocabulary to negative infinity."""
    if unpadded_vocab_size is None or unpadded_vocab_size >= vocab_size:
        return logits
    device = logits.device
    # Two broadcast scalars rather than a materialized row: keeps a
    # vocab-sized fp32 constant out of the graph.
    keep = F.broadcast_to(
        F.constant(0.0, DType.float32, device=device),
        shape=[unpadded_vocab_size],
    )
    drop = F.broadcast_to(
        F.constant(float("-inf"), DType.float32, device=device),
        shape=[vocab_size - unpadded_vocab_size],
    )
    return logits + F.cast(F.concat([keep, drop]), logits.dtype)


class DeepseekV3_2TextModel(
    Module[
        [
            Tensor,
            PagedCacheValues,
            PagedCacheValues,
            Tensor,
            Tensor,
            Tensor,
            Tensor | None,
            Tensor | None,
            EPCommBuffers | None,
        ],
        ModuleV3Outputs,
    ]
):
    """The DeepseekV3.2 language model.

    DeepseekV3 with sparse attention: a lightning indexer selects the keys each
    query attends to, backed by its own FP8 key cache.
    """

    def __init__(
        self,
        config: DeepseekV3_2Config,
        ep_batch_manager: EPBatchManager | None = None,
    ) -> None:
        self.ep_batch_manager = ep_batch_manager

        self.rope: RotaryEmbedding
        if config.rope_scaling is not None:
            scaling_params = DeepseekYarnRopeScalingParams(
                scaling_factor=config.rope_scaling["factor"],
                original_max_position_embeddings=config.rope_scaling[
                    "original_max_position_embeddings"
                ],
                beta_fast=config.rope_scaling["beta_fast"],
                beta_slow=config.rope_scaling["beta_slow"],
                mscale=config.rope_scaling["mscale"],
                mscale_all_dim=config.rope_scaling["mscale_all_dim"],
            )
            self.rope = DeepseekYarnRotaryEmbedding(
                dim=config.qk_rope_head_dim,
                n_heads=config.num_attention_heads,
                theta=config.rope_theta,
                max_seq_len=config.max_position_embeddings,
                device=config.devices[0].to_device(),
                interleaved=config.rope_interleave,
                scaling_params=scaling_params,
            )
        else:
            # GLM-5.x declares rope_type "default"; only DeepSeek proper uses
            # YaRN. Sized by ``max_seq_len`` rather than the model's
            # ``max_position_embeddings`` (1M for GLM): plain RoPE frequencies
            # depend only on position, so the shorter table is equivalent and
            # avoids materializing hundreds of MB of unused rows.
            self.rope = RotaryEmbedding(
                dim=config.qk_rope_head_dim,
                n_heads=config.num_attention_heads,
                theta=config.rope_theta,
                max_seq_len=config.max_seq_len,
                device=config.devices[0].to_device(),
                head_dim=config.qk_rope_head_dim,
                interleaved=config.rope_interleave,
            )

        # Override the tensor parallel axis if data parallelism is enabled.
        tp_axis = DP if config.data_parallel_degree > 1 else None
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            dim=config.hidden_size,
            tp_axis=tp_axis,
        )

        self.norm = MultiplyBeforeCastRMSNorm(
            dim=config.hidden_size, eps=config.rms_norm_eps
        )

        self.lm_head = ColumnParallelLinear(
            in_dim=config.hidden_size,
            out_dim=config.vocab_size,
            bias=False,
            tp_axis=tp_axis,
        )

        qk_head_dim = config.qk_rope_head_dim + config.qk_nope_head_dim
        scale = self.rope.compute_scale(math.sqrt(1.0 / qk_head_dim))
        layers = []
        for i in range(config.num_hidden_layers):
            layers.append(
                DeepseekV3_2TransformerBlock(
                    config=config,
                    layer_idx=i,
                    attention_scale=scale,
                    ep_batch_manager=ep_batch_manager
                    if i >= config.first_k_dense_replace
                    else None,
                )
            )

        self.dim = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.layers = ModuleList[DeepseekV3_2TransformerBlock](layers)
        self.kv_params = config.kv_params
        self.config = config
        self.mesh = config.mesh

    def forward(
        self,
        tokens: Tensor,
        kv_collection: PagedCacheValues,
        indexer_kv_collection: PagedCacheValues,
        return_n_logits: Tensor,
        input_row_offsets: Tensor,
        batch_context_length: Tensor,
        data_parallel_splits: Tensor | None = None,
        input_row_offsets_i64: Tensor | None = None,
        comm_buffers: EPCommBuffers | None = None,
    ) -> ModuleV3Outputs:
        if self.mesh is not None:
            if self.mesh.num_devices > 1:
                tokens = F.distributed_broadcast(tokens, self.mesh)
                input_row_offsets = F.distributed_broadcast(
                    input_row_offsets, self.mesh
                )
            else:
                # The broadcast collective needs signal buffers, which are
                # only allocated for multi-device runs. Onto a single-device
                # mesh, replicating is just a placement change.
                tokens = tokens.to(self.mesh)
                input_row_offsets = input_row_offsets.to(self.mesh)

        h = self.embed_tokens(tokens)
        if self.config.data_parallel_degree > 1:
            assert data_parallel_splits is not None
            assert input_row_offsets_i64 is not None
            assert self.mesh is not None
            batch_placement = tuple(
                Sharded(0) if name == "dp" else Replicated()
                for name in self.mesh.axis_names
            )
            h, input_row_offsets = split_replicated_batch(
                h,
                input_row_offsets,
                input_row_offsets_i64,
                data_parallel_splits,
                DeviceMapping(self.mesh, batch_placement),
            )

        freqs_cis = F.cast(self.rope.freqs_cis, h.dtype)
        if self.mesh is not None:
            freqs_cis = freqs_cis.to(self.mesh)
        else:
            freqs_cis = freqs_cis.to(h.device)

        # The MLA prefill plan depends only on the sequence layout (identical
        # across layers), so compute it once and thread it into every layer
        # instead of recomputing it per layer. Decode has no prefill plan.
        mla_prefill_metadata = None
        first_attn = self.layers[0].self_attn
        if first_attn.graph_mode in ("prefill", "auto"):
            mla_prefill_metadata = first_attn.create_mla_prefill_metadata(
                input_row_offsets, kv_collection
            )
            # Host-substitute the per-layer D2H buffer_length copies with the
            # CPU batch_context_length so the graph stays capturable.
            mla_prefill_metadata.buffer_lengths = batch_context_length

        # MoE blocks share a subgraph, as in V3. ``full`` and ``shared``
        # layers differ structurally (shared layers carry no indexer weights),
        # so they get one group each: a shared subgraph name is what makes two
        # call sites reuse a body, and the cache key does not otherwise see
        # the difference.
        topk_indices: Tensor | None = None
        for idx, layer in enumerate(self.layers):
            layer_idx_tensor = F.constant(idx, DType.uint32, device=CPU())
            call: Callable[..., tuple[Tensor, Tensor]] = layer
            if isinstance(layer.mlp, QuantizedMoE):
                group = "shared" if layer.self_attn.skip_topk else "full"
                call = as_subgraph(layer, name=f"moe_block_{group}")
            h, topk_indices = call(
                layer_idx_tensor,
                h,
                kv_collection,
                indexer_kv_collection,
                input_row_offsets,
                freqs_cis,
                mla_prefill_metadata,
                comm_buffers,
                topk_indices,
                False,
            )

        last_token_h = gather_last_tokens(h, input_row_offsets)
        if self.config.data_parallel_degree > 1:
            last_token_h = F.allgather(last_token_h)
        last_logits = self.lm_head(self.norm(last_token_h))
        if self.mesh is not None:
            last_logits = last_logits.to(self.mesh.devices[0])
        last_logits = F.cast(last_logits, DType.float32)
        last_logits = mask_padded_tail(
            last_logits,
            self.config.vocab_size,
            self.config.unpadded_vocab_size,
        )
        return ModuleV3Outputs(next_token_logits=last_logits)


class DeepseekV3_2(Module[..., ModuleV3Outputs]):
    """Top-level DeepseekV3.2 wrapper that unflattens variadic KV cache args."""

    def __init__(
        self,
        config: DeepseekV3_2Config,
        kv_params: KVCacheParamInterface,
        ep_batch_manager: EPBatchManager | None = None,
    ) -> None:
        super().__init__()
        self.language_model = DeepseekV3_2TextModel(config, ep_batch_manager)
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
        mesh = self.config.mesh
        assert mesh is not None

        # Reconstruct inputs from variadic arguments.
        data_parallel_splits: Tensor | None = None
        input_row_offsets_i64: Tensor | None = None
        dp_degree = self.config.data_parallel_degree
        batch_context_lengths = variadic_args[:dp_degree]
        variadic_args = variadic_args[dp_degree:]
        if dp_degree > 1:
            assert dp_degree == mesh.num_devices

            cpu_mesh = DeviceMesh(
                tuple(CPU() for _ in range(dp_degree)), (dp_degree,), ("dp",)
            )
            batch_context_lengths_tensor = Tensor.from_shard_values(
                [TensorValue(shard) for shard in batch_context_lengths],
                DeviceMapping(cpu_mesh, (Replicated(),) * mesh.ndim),
            )

            data_parallel_splits, input_row_offsets_i64, *rest = variadic_args
            variadic_args = tuple(rest)
        else:
            batch_context_lengths_tensor = batch_context_lengths[0]

        kv_inputs = iter(x._graph_value for x in variadic_args)
        # A ``MultiKVCacheParams`` tree unflattens to a dict keyed by child
        # name, so the two caches are addressed by name rather than position.
        kv_tree = self.kv_params.unflatten_kv_inputs(kv_inputs)
        assert isinstance(kv_tree, dict)

        # Replicas serve different requests, so their caches vary;
        # tensor-parallel devices all store the same latent cache.
        kv_mapping = DeviceMapping(
            mesh,
            tuple(
                Unknown() if name == DP else Replicated()
                for name in mesh.axis_names
            ),
        )

        def collection(name: str) -> PagedCacheValues:
            leaves = tree.leaves(kv_tree[name], leaf=KVCacheInputsPerDevice)
            return PagedCacheValues.from_upstream(leaves, kv_mapping)

        kv_collection = collection("mla")
        indexer_kv_collection = collection("indexer")

        comm_buffers: EPCommBuffers | None = None
        if self.ep_batch_manager is not None:
            # Any variadic graph values left after the KV cache are the EP
            # communication buffers. Wrap them as an explicit forward argument
            # so they thread through each MoE block's subgraph boundary.
            ep_buffers = list(kv_inputs)
            comm_buffers = self.ep_batch_manager.comm_buffers(ep_buffers)
        return self.language_model(
            tokens,
            kv_collection,
            indexer_kv_collection,
            return_n_logits,
            input_row_offsets,
            batch_context_lengths_tensor,
            data_parallel_splits,
            input_row_offsets_i64,
            comm_buffers,
        )
