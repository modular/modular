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
"""Implements the DeepseekV3 model using the ModuleV3 API."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import ClassVar

from max import functional as F
from max import tree
from max.driver import CPU
from max.dtype import DType
from max.graph import TensorValue, ops
from max.nn import Module, as_subgraph
from max.nn.common_layers.embedding import VocabParallelEmbedding
from max.nn.common_layers.kv_cache import (
    PagedCacheValues,
    cache_tensor,
    kv_input_values,
)
from max.nn.common_layers.linear import ColumnParallelLinear
from max.nn.common_layers.mesh_axis import DP
from max.nn.common_layers.quant_moe import (
    QuantizedMoE,
)
from max.nn.ep import EPBuffers
from max.nn.kv_cache import KVCacheInputsPerDevice
from max.nn.norm import RMSNorm
from max.nn.rope_scaling import DeepseekYarnRopeScalingParams
from max.nn.sequential import ModuleList
from max.pipelines.lib import ModuleV3Outputs
from max.pipelines.lib.role_contracts import DecoderInputs, DecoderOutputs
from max.pipelines.lib.utils import split_replicated_batch
from max.tensor import Tensor, default_dtype
from max.tensor.sharding import (
    DeviceMapping,
    DeviceMesh,
    Replicated,
    Sharded,
    Unknown,
)

from ..deepseekV2_modulev3.layers.rotary_embedding import (
    DeepseekYarnRotaryEmbedding,
)
from .layers.transformer_block import DeepseekV3TransformerBlock
from .model_config import DeepseekV3Config


def gather_last_tokens(h: Tensor, input_row_offsets: Tensor) -> Tensor:
    """Gathers the last token of each request from the ragged batch, per shard.

    Under data parallelism ``input_row_offsets`` is rebased into each replica's
    local frame, so its values only address that replica's own rows; the gather
    runs on each device's own rows, never through the auto-sharded
    ``F.gather``.
    """

    def _body(h: TensorValue, offsets: TensorValue) -> TensorValue:
        return ops.gather(h, offsets[1:] - 1, axis=0)

    if not h.is_distributed:
        return F.gather(h, input_row_offsets[1:] - 1, axis=0)
    # The rule-less body runs on each device's own rows; its result keeps h's
    # placements.
    return F.functional(_body)(h, input_row_offsets).rebind_mapping(h.mapping)


class DeepseekV3TextModel(
    Module[
        [
            Tensor,
            PagedCacheValues,
            Tensor,
            Tensor,
            Tensor,
            Tensor | None,
            Tensor | None,
            EPBuffers | None,
        ],
        ModuleV3Outputs,
    ]
):
    """The DeepseekV3 language model.

    Decoder-only Transformer with Multi-Latent Attention, MoE feed-forward
    (using a noaux_tc sigmoid router), and DeepSeek YaRN rotary embeddings.
    """

    def __init__(self, config: DeepseekV3Config) -> None:
        assert config.rope_scaling is not None
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

        # Override the tensor parallel axis if data parallelism is enabled.
        tp_axis = DP if config.data_parallel_degree > 1 else None
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            dim=config.hidden_size,
            tp_axis=tp_axis,
        )

        self.norm = RMSNorm(dim=config.hidden_size, eps=config.rms_norm_eps)

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
                DeepseekV3TransformerBlock(
                    config=config,
                    layer_idx=i,
                    attention_scale=scale,
                )
            )

        self.dim = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.layers = ModuleList[DeepseekV3TransformerBlock](layers)
        self.kv_params = config.kv_params
        self.config = config
        self.mesh = config.mesh

    def forward(
        self,
        tokens: Tensor,
        kv_collection: PagedCacheValues,
        return_n_logits: Tensor,
        input_row_offsets: Tensor,
        batch_context_length: Tensor,
        data_parallel_splits: Tensor | None = None,
        input_row_offsets_i64: Tensor | None = None,
        ep: EPBuffers | None = None,
    ) -> ModuleV3Outputs:
        """Runs the embedding and decoder stack over the tokens."""
        if self.mesh is not None:
            tokens = F.distributed_broadcast(tokens, self.mesh)
            input_row_offsets = F.distributed_broadcast(
                input_row_offsets, self.mesh
            )

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

        # All MoE blocks share one subgraph, built from the first block. Each
        # call resolves the weights of its own layer.
        for idx, layer in enumerate(self.layers):
            layer_idx_tensor = F.constant(idx, DType.uint32, device=CPU())
            call: Callable[..., Tensor] = layer
            if isinstance(layer.mlp, QuantizedMoE):
                call = as_subgraph(layer, name="moe_block")
            h = call(
                layer_idx_tensor,
                h,
                kv_collection,
                input_row_offsets,
                freqs_cis,
                mla_prefill_metadata,
                ep,
            )

        last_token_h = gather_last_tokens(h, input_row_offsets)
        if self.config.data_parallel_degree > 1:
            last_token_h = F.allgather(last_token_h)
        last_logits = self.lm_head(self.norm(last_token_h))
        if self.mesh is not None:
            last_logits = last_logits.to(self.mesh.devices[0])
        last_logits = F.cast(last_logits, DType.float32)
        return ModuleV3Outputs(next_token_logits=last_logits)


class DeepseekV3Decoder(Module[[DecoderInputs], DecoderOutputs]):
    """The DeepSeek-V3 MLA decoder, registered for the ``DECODER`` role.

    It runs tensor-parallel attention over a ``tp`` mesh, or one attention
    replica per device over a ``dp`` mesh, with its experts tensor-parallel
    or, under expert parallelism, spread across the devices.
    """

    graph_name: ClassVar[str] = "DeepseekV3"
    supports_data_parallel: ClassVar[bool] = True

    def __init__(self, config: DeepseekV3Config) -> None:
        super().__init__()
        if config.mesh is None:
            raise ValueError(
                "DeepSeek-V3 needs a mesh: apply the runtime first"
            )
        # Quantized checkpoints keep norms, biases and embeddings in bf16.
        dtype = (
            DType.bfloat16 if config.quant_config is not None else config.dtype
        )
        with default_dtype(dtype):
            self.language_model = DeepseekV3TextModel(config)
        self.config = config
        self.kv_params = config.kv_params

    def forward(self, inputs: DecoderInputs) -> DecoderOutputs:
        """Runs the decoder over the batch."""
        mesh = self.config.mesh
        assert mesh is not None
        assert inputs.batch_context_lengths is not None

        data_parallel_splits: Tensor | None = None
        input_row_offsets_i64: Tensor | None = None
        dp_degree = self.config.data_parallel_degree
        if dp_degree > 1:
            assert dp_degree == mesh.num_devices
            assert inputs.host_input_row_offsets is not None
            cpu_mesh = DeviceMesh(
                tuple(CPU() for _ in range(dp_degree)), (dp_degree,), ("dp",)
            )
            batch_context_length = cache_tensor(
                inputs.batch_context_lengths,
                DeviceMapping(cpu_mesh, (Replicated(),) * mesh.ndim),
            )
            data_parallel_splits = inputs.data_parallel_splits
            input_row_offsets_i64 = F.cast(
                inputs.host_input_row_offsets, DType.int64
            )
        else:
            batch_context_length = inputs.batch_context_lengths[0]

        kv_inputs = iter(kv_input_values(inputs.kv))
        kv_collections = tree.leaves(
            self.kv_params.unflatten_kv_inputs(kv_inputs),
            leaf=KVCacheInputsPerDevice,
        )
        # Replicas serve different requests, so their caches vary;
        # tensor-parallel devices all store the same latent cache.
        kv_collection = PagedCacheValues.from_upstream(
            kv_collections,
            DeviceMapping(
                mesh,
                tuple(
                    Unknown() if name == DP else Replicated()
                    for name in mesh.axis_names
                ),
            ),
        )

        return self.language_model(
            inputs.tokens,
            kv_collection,
            inputs.return_n_logits,
            inputs.input_row_offsets,
            batch_context_length,
            data_parallel_splits,
            input_row_offsets_i64,
            inputs.ep,
        )
