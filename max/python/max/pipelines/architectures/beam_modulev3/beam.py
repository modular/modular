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
"""Implements the Beam model on ModuleV3.

Layers that agree on attention type, MLP type and MLP precision share one
traced subgraph. Everything else that differs between the layers of a group
reaches the subgraph as an operand.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import ClassVar

from max import functional as F
from max import tree
from max.driver import CPU
from max.dtype import DType
from max.graph import DimLike
from max.nn import Module, as_subgraph
from max.nn.common_layers.kv_cache import PagedCacheValues
from max.nn.common_layers.linear import ColumnParallelLinear
from max.nn.common_layers.rotary_embedding import RotaryEmbedding
from max.nn.embedding import Embedding
from max.nn.kv_cache import KVCacheParams, MultiKVCacheParams
from max.nn.legacy.comm.ep import EPBatchManager
from max.nn.outputs import ReturnLogits
from max.nn.sequential import ModuleList
from max.pipelines.architectures.gemma4_modulev3.layers.rms_norm import (
    Gemma4RMSNorm,
)
from max.pipelines.lib.role_contracts import DecoderInputs, DecoderOutputs
from max.tensor import Tensor, default_dtype
from max.tensor.sharding import DeviceMapping

from .layers.transformer_block import BeamTransformerBlock, residual_scales
from .model_config import BeamConfig


def nope_query_scale(
    input_row_offsets: Tensor, cache_lengths: Tensor, total_seq_len: DimLike
) -> Tensor:
    """Returns the float32 per-token query scale of the NoPE layers.

    That is ``(1 + (position + 1) / 2048) ** 0.1``, evaluated as the reference
    does, over each token's absolute position: its index in its sequence
    plus the tokens its sequence already has cached.
    """
    offsets = input_row_offsets.cast(DType.int64)
    starts = offsets[:-1]
    shift = F.rebind(cache_lengths.cast(DType.int64), starts.shape) - starts
    positions = F.range(
        0,
        F.shape_to_tensor([total_seq_len])[0],
        1,
        out_dim=total_seq_len,
        dtype=DType.int64,
        device=offsets.mapping,
    ) + F.repeat_interleave(
        shift, offsets[1:] - starts, axis=0, out_dim=total_seq_len
    )
    return F.exp(0.1 * F.log1p((positions.cast(DType.float32) + 1.0) / 2048.0))


class Beam(Module[[DecoderInputs], DecoderOutputs]):
    """The Beam decoder stack, embeddings and LM head.

    Attention heads are tensor parallel over the mesh, and with more than
    one device the routed experts are expert parallel over all of them: the
    attention output is reduce-scattered over the tokens and each device
    routes its own rows. The routed experts are NVFP4 on a quantized
    checkpoint; every other layer is bfloat16.

    Args:
        config: The Beam config, with the runtime applied.
    """

    input_layout: ClassVar[tuple[str, ...]] = (
        "tokens",
        "return_n_logits",
        "input_row_offsets",
        "kv",
        "ep",
    )
    """The graph's input order from before registration, which exported
    artifacts and Mach's registry entry bind."""

    def __init__(self, config: BeamConfig) -> None:
        super().__init__()
        if config.mesh is None:
            raise ValueError("Beam needs a mesh: apply the runtime first")
        return_logits = config.return_logits
        if return_logits not in (ReturnLogits.LAST_TOKEN, ReturnLogits.ALL):
            raise NotImplementedError(
                f"Beam does not return {return_logits} logits."
            )
        kv_params = config.kv_params
        assert isinstance(kv_params, MultiKVCacheParams)
        ep_batch_manager = (
            None
            if config.ep_config is None
            else EPBatchManager(config.ep_config)
        )
        self.return_logits = return_logits
        self.kv_params = kv_params
        self.mesh = config.mesh
        self.ep_batch_manager = ep_batch_manager
        self.use_subgraphs = config.use_subgraphs
        self.logit_softcapping = config.final_logit_softcapping

        # On an NVFP4 checkpoint ``config.dtype`` is the packed storage dtype;
        # everything outside the routed experts is bfloat16.
        with default_dtype(DType.bfloat16):
            self.rope = RotaryEmbedding(
                dim=config.hidden_size,
                n_heads=config.num_attention_heads,
                theta=config.rope_theta,
                max_seq_len=config.max_seq_len,
                device=CPU(),
                head_dim=config.head_dim,
                interleaved=False,
            )
            self.embed_tokens = Embedding(
                config.vocab_size, dim=config.hidden_size
            )
            self.norm = Gemma4RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
            self.lm_head = ColumnParallelLinear(
                config.hidden_size, config.vocab_size, bias=False
            )

            quant_config = config.quant_config
            quantized = (
                quant_config.mlp_quantized_layers if quant_config else ()
            )
            counts = dict.fromkeys(kv_params.children, 0)
            self._layer_idx_in_cache: list[int] = []
            self._subgraph_names: list[str] = []
            layers = []
            for i, layer_type in enumerate(config.layer_types):
                self._layer_idx_in_cache.append(counts[layer_type])
                counts[layer_type] += 1
                mlp_type = config.mlp_layer_types[i]
                precision = "_nvfp4" if i in quantized else ""
                self._subgraph_names.append(
                    f"beam_{layer_type}_{mlp_type}{precision}_block"
                )
                leaf = kv_params.children[layer_type]
                assert isinstance(leaf, KVCacheParams)
                layers.append(
                    BeamTransformerBlock(config, i, leaf, ep_batch_manager)
                )
            self.layers = ModuleList(layers)
        self._layer_types = list(config.layer_types)

    def forward(self, inputs: DecoderInputs) -> DecoderOutputs:
        """Runs the model and returns float32 logits."""
        kv_inputs = iter(x._graph_value for x in tree.leaves(inputs.kv))
        kv_leaves = self.kv_params.unflatten_basic_kv_tree(kv_inputs)
        comm_buffers = (
            self.ep_batch_manager.comm_buffers(
                [x._graph_value for x in tree.leaves(inputs.ep)]
            )
            if self.ep_batch_manager is not None
            else None
        )
        # Each device owns a whole cache, so the cache inputs are replicated
        # over the mesh.
        kv_mapping = DeviceMapping.replicated(self.mesh)
        kv_by_type = {
            layer_type: PagedCacheValues.from_upstream(leaf, kv_mapping)
            for layer_type, leaf in zip(
                self.kv_params.children, kv_leaves, strict=True
            )
        }
        tokens = F.distributed_broadcast(inputs.tokens, self.mesh)
        input_row_offsets = F.distributed_broadcast(
            inputs.input_row_offsets, self.mesh
        )

        h = self.embed_tokens(tokens)
        position_inputs = {
            "sliding_attention": self.rope.freqs_cis.to(kv_mapping),
            "full_attention": nope_query_scale(
                input_row_offsets,
                kv_by_type["full_attention"].cache_lengths,
                tokens.shape[0],
            ),
        }
        for idx, layer in enumerate(self.layers):
            layer_type = self._layer_types[idx]
            call: Callable[..., Tensor] = layer
            if self.use_subgraphs:
                call = as_subgraph(layer, name=self._subgraph_names[idx])
            h = call(
                F.constant(
                    self._layer_idx_in_cache[idx], DType.uint32, device=CPU()
                ),
                h,
                kv_by_type[layer_type],
                position_inputs[layer_type],
                input_row_offsets,
                F.constant(
                    residual_scales(idx), DType.float32, device=kv_mapping
                ),
                comm_buffers,
            )

        # The logits are replicated after the vocabulary gather, and the
        # sampler reads device 0's copy.
        last_logits = self._logits(
            F.gather(h, input_row_offsets[1:] - 1, axis=0)
        ).local_shards[0]
        if self.return_logits == ReturnLogits.ALL:
            return DecoderOutputs(
                next_token_logits=last_logits,
                logits=self._logits(h).local_shards[0],
                logit_offsets=input_row_offsets.local_shards[0],
            )
        return DecoderOutputs(next_token_logits=last_logits)

    def _logits(self, h: Tensor) -> Tensor:
        logits = F.allgather(self.lm_head(self.norm(h)), tensor_axis=-1)
        logits = logits.cast(DType.float32)
        if self.logit_softcapping:
            cap = self.logit_softcapping
            logits = F.tanh(logits / cap) * cap
        return logits
