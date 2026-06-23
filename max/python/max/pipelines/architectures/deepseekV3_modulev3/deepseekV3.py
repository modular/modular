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

from max.driver import CPU
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Module
from max.experimental.nn.common_layers.embedding import VocabParallelEmbedding
from max.experimental.nn.common_layers.kv_cache import PagedCacheValues
from max.experimental.nn.common_layers.linear import ColumnParallelLinear
from max.experimental.nn.norm import RMSNorm
from max.experimental.nn.sequential import ModuleList
from max.experimental.sharding import (
    NoReshard,
    PlacementMapping,
    Replicated,
    mode,
)
from max.experimental.tensor import Tensor
from max.nn.comm.ep import EPBatchManager
from max.nn.kv_cache import (
    KVCacheInputs,
    KVCacheParamInterface,
)
from max.nn.rotary_embedding import DeepseekYarnRopeScalingParams

from ..deepseekV2_modulev3.layers.rotary_embedding import (
    DeepseekYarnRotaryEmbedding,
)
from .layers.transformer_block import DeepseekV3TransformerBlock
from .model_config import DeepseekV3Config


class DeepseekV3TextModel(
    Module[[Tensor, PagedCacheValues, Tensor, Tensor], tuple[Tensor, ...]]
):
    """The DeepseekV3 language model.

    Decoder-only Transformer with Multi-Latent Attention, MoE feed-forward
    (using a noaux_tc sigmoid router), and DeepSeek YaRN rotary embeddings.
    """

    def __init__(
        self,
        config: DeepseekV3Config,
        ep_batch_manager: EPBatchManager | None = None,
    ) -> None:
        assert config.rope_scaling is not None
        self.ep_batch_manager = ep_batch_manager

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

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            dim=config.hidden_size,
        )

        self.norm = RMSNorm(dim=config.hidden_size, eps=config.rms_norm_eps)

        self.lm_head = ColumnParallelLinear(
            in_dim=config.hidden_size,
            out_dim=config.vocab_size,
            bias=False,
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
                    ep_batch_manager=ep_batch_manager,
                )
            )

        self.dim = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.layers = ModuleList(layers)
        self.kv_params = config.kv_params
        self.config = config
        self.mesh = config.mesh

    @mode(NoReshard())
    def forward(
        self,
        tokens: Tensor,
        kv_collection: PagedCacheValues,
        return_n_logits: Tensor,
        input_row_offsets: Tensor,
    ) -> tuple[Tensor, ...]:
        if self.mesh is not None:
            tokens = tokens.to(self.mesh)
            input_row_offsets = input_row_offsets.to(self.mesh)

        h = self.embed_tokens(tokens)

        freqs_cis = F.cast(self.rope.freqs_cis, h.dtype)
        if self.mesh is not None:
            freqs_cis = freqs_cis.to(self.mesh)
        else:
            freqs_cis = freqs_cis.to(h.device)

        for idx, layer in enumerate(self.layers):
            layer_idx_tensor = F.constant(idx, DType.uint32, device=CPU())
            h = layer(
                layer_idx_tensor,
                h,
                kv_collection,
                input_row_offsets,
                freqs_cis,
            )

        last_token_indices = input_row_offsets[1:] - 1
        last_token_h = F.gather(h, last_token_indices, axis=0)
        last_logits = F.cast(
            self.lm_head(self.norm(last_token_h)),
            DType.float32,
        )
        # The logits are replicated across the mesh; collapse to a single
        # device so the pipeline can read them as one buffer.
        if self.mesh is not None:
            last_logits = last_logits.to(self.mesh.devices[0])
        return (last_logits,)


class DeepseekV3(Module[..., tuple[Tensor, ...]]):
    """Top-level DeepseekV3 wrapper that unflattens variadic KV cache args."""

    def __init__(
        self,
        config: DeepseekV3Config,
        kv_params: KVCacheParamInterface,
        ep_batch_manager: EPBatchManager | None = None,
    ) -> None:
        super().__init__()
        self.language_model = DeepseekV3TextModel(config, ep_batch_manager)
        self.config = config
        self.kv_params = kv_params
        self.ep_batch_manager = ep_batch_manager

    def forward(
        self,
        tokens: Tensor,
        return_n_logits: Tensor,
        input_row_offsets: Tensor,
        *variadic_args: Tensor,
    ) -> tuple[Tensor, ...]:
        kv_inputs = iter(x._graph_value for x in variadic_args)
        kv_collections = self.kv_params.unflatten_kv_inputs(kv_inputs)
        assert isinstance(kv_collections, KVCacheInputs)

        # Any variadic graph values left after the KV cache are the EP
        # communication buffers; hand them to the batch manager so the MoE
        # dispatch/combine kernels can reference them.
        if self.ep_batch_manager is not None:
            self.ep_batch_manager.fetch_buffers(list(kv_inputs))
        # Combine the per-device upstream KV collections into a single
        # mesh-distributed PagedCacheValues (one shard per device).
        mesh = self.config.mesh
        if mesh is not None:
            kv_collection = PagedCacheValues.from_upstream(
                kv_collections.inputs,
                PlacementMapping(mesh, (Replicated(),) * mesh.ndim),
            )
        else:
            raise ValueError("Mesh must be define")
        return self.language_model(
            tokens, kv_collection, return_n_logits, input_row_offsets
        )
