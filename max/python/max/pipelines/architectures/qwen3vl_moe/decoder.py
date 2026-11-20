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
"""The Qwen3-VL decoder, dense or mixture of experts, with deepstack."""

from __future__ import annotations

from typing import ClassVar

from max import functional as F
from max import tree
from max.driver import CPU
from max.dtype import DType
from max.graph import DeviceRef, TensorType
from max.nn import Linear, Module
from max.nn.common_layers.embedding import VocabParallelEmbedding
from max.nn.common_layers.kv_cache import PagedCacheValues, kv_input_values
from max.nn.common_layers.linear import col_parallel
from max.nn.common_layers.quant_linear import (
    QuantizedMLP,
    tensor_parallel_mlp,
)
from max.nn.kv_cache import KVCacheInputsPerDevice, KVCacheParams
from max.nn.outputs import ReturnLogits
from max.nn.sequential import ModuleList
from max.pipelines.lib import ModuleV3Outputs
from max.pipelines.lib.role_contracts import (
    DecoderInputs,
    DecoderOutputs,
    VisionInputs,
)
from max.pipelines.lib.vlm_utils import F_merge_multimodal_embeddings
from max.tensor import Tensor, default_dtype
from max.tensor.sharding import DeviceMapping, DeviceMesh

from .layers.attention import Qwen3VLAttention
from .layers.moe import Qwen3VLMoE
from .layers.norm import RMSNorm
from .layers.rotary import mrope_freqs_cis
from .model_config import Qwen3VLConfig

__all__ = ["Qwen3VLDecoder", "Qwen3VLTextModel"]


def _replicate(value: Tensor, mesh: DeviceMesh) -> Tensor:
    """Places a single-device tensor on every device of ``mesh``.

    A broadcast collective, not a peer copy, which device graph capture
    cannot record. Onto one device it is a placement.
    """
    if mesh.num_devices > 1:
        return F.distributed_broadcast(value, mesh)
    return value.to(mesh)


class Qwen3VLDecoderLayer(Module[..., Tensor]):
    """A pre-norm block: attention, then the MLP or the routed experts."""

    def __init__(self, config: Qwen3VLConfig, layer_idx: int) -> None:
        quant_config = config.quant_config
        norm_dtype = config.norm_dtype or config.compute_dtype
        assert isinstance(config.kv_params, KVCacheParams)
        self.self_attn = Qwen3VLAttention(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            kv_params=config.kv_params,
            norm_eps=config.rms_norm_eps,
            quant_config=quant_config,
        )
        self.mlp: Qwen3VLMoE | QuantizedMLP
        if config.is_moe_layer(layer_idx):
            self.mlp = Qwen3VLMoE(
                hidden_dim=config.hidden_size,
                num_experts=config.num_experts,
                num_experts_per_token=config.num_experts_per_tok,
                moe_dim=config.moe_intermediate_size,
                quant_config=quant_config,
            )
        else:
            self.mlp = tensor_parallel_mlp(
                QuantizedMLP(
                    config.hidden_size,
                    config.intermediate_size,
                    quant_config=quant_config,
                )
            )
        self.input_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, dtype=norm_dtype
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, dtype=norm_dtype
        )

    def forward(
        self,
        h: Tensor,
        kv_collection: PagedCacheValues,
        layer_idx: Tensor,
        freqs_cis: Tensor,
        input_row_offsets: Tensor,
    ) -> Tensor:
        """Runs the block; the partial sums reduce before each residual add."""
        attn = self.self_attn(
            self.input_layernorm(h),
            kv_collection,
            layer_idx,
            freqs_cis,
            input_row_offsets,
        )
        h = h + attn.to(h.mapping)
        return h + self.mlp(self.post_attention_layernorm(h)).to(h.mapping)


class Qwen3VLTextModel(Module[..., ModuleV3Outputs]):
    """The embedding, the layer stack, the final norm and the head.

    The first layers' outputs also take the deepstack image rows the tower
    produced from its intermediate blocks.
    """

    def __init__(self, config: Qwen3VLConfig) -> None:
        self.config = config
        self.num_deepstack = len(config.vision_config.deepstack_visual_indexes)
        self.layers = ModuleList(
            [
                Qwen3VLDecoderLayer(config, i)
                for i in range(config.num_hidden_layers)
            ]
        )
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, dim=config.hidden_size
        )
        self.norm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.norm_dtype or config.compute_dtype,
        )
        self.lm_head: Linear | None = None
        if not config.tie_word_embeddings:
            # Column-parallel: each device projects onto its share of the
            # vocabulary.
            self.lm_head = col_parallel(
                Linear(config.hidden_size, config.vocab_size, bias=False)
            )

    def head(self, x: Tensor) -> Tensor:
        """Returns the head's logits of ``x``, gathered on the first device."""
        if self.lm_head is None:
            logits = x @ self.embed_tokens.weight.T
        else:
            logits = self.lm_head(x)
        return F.allgather(logits, tensor_axis=-1).local_shards[0]

    def logits(self, h: Tensor) -> Tensor:
        """Returns float32 logits of the normed ``h``, on the first device."""
        return self.head(self.norm(h)).cast(DType.float32)

    @staticmethod
    def window_rows(input_row_offsets: Tensor, count: Tensor) -> Tensor:
        """Returns the rows of each sequence's last ``count`` tokens."""
        distance = F.range(
            count,
            0,
            -1,
            out_dim="return_n_logits_range",
            dtype=DType.int64,
            device=input_row_offsets.mesh,
        )
        return (input_row_offsets[1:].unsqueeze(-1) - distance).reshape([-1])

    def forward(
        self,
        tokens: Tensor,
        input_row_offsets: Tensor,
        return_n_logits: Tensor,
        kv_collection: PagedCacheValues,
        vision: VisionInputs[Tensor],
        position_ids: Tensor,
    ) -> ModuleV3Outputs:
        """Runs the decoder and returns the logits it was built to.

        Args:
            tokens: The ragged token ids, replicated on the mesh.
            input_row_offsets: The ragged row offsets, replicated.
            return_n_logits: How many logits per sequence, on the host.
            kv_collection: The decoder's cache.
            vision: The image rows, each its merged embedding then its
                deepstack embeddings, and their token positions.
            position_ids: ``[3, total_seq_len]`` M-RoPE positions on the
                host.
        """
        config = self.config
        mesh = tokens.mesh
        hidden = config.hidden_size
        rows = vision.embeddings
        indices = vision.scatter_indices
        h = F_merge_multimodal_embeddings(
            self.embed_tokens(tokens), rows[:, :hidden], indices
        )
        freqs_cis = mrope_freqs_cis(
            position_ids,
            theta=config.rope_theta,
            head_dim=config.head_dim,
            mrope_section=config.mrope_section,
            dtype=DType.bfloat16,
        ).to(DeviceMapping.replicated(mesh))
        for index, layer in enumerate(self.layers):
            h = layer(
                h,
                kv_collection,
                F.constant(index, DType.uint32, device=CPU()),
                freqs_cis,
                input_row_offsets,
            )
            if index < self.num_deepstack:
                start = hidden * (index + 1)
                deepstack = rows[:, start : start + hidden]
                zeros = F.broadcast_to(
                    F.constant(0, h.dtype, device=h.mapping), h.shape
                )
                h = h + F_merge_multimodal_embeddings(zeros, deepstack, indices)

        last = self.logits(F.gather(h, input_row_offsets[1:] - 1, axis=0))
        logits: Tensor | None = None
        offsets: Tensor | None = None
        if config.return_logits == ReturnLogits.ALL:
            logits = self.logits(h)
            offsets = input_row_offsets.local_shards[0]
        elif config.return_logits == ReturnLogits.VARIABLE:
            count = return_n_logits[0]
            window = self.window_rows(input_row_offsets, count)
            logits = self.logits(F.gather(h, window, axis=0))
            offsets = F.range(
                0,
                window.shape[0] + count,
                count,
                out_dim="logit_offsets",
                dtype=DType.int64,
                device=tokens.local_shards[0].device,
            )
        return ModuleV3Outputs(
            next_token_logits=last, logits=logits, logit_offsets=offsets
        )


class Qwen3VLDecoder(Module[[DecoderInputs[Tensor]], DecoderOutputs]):
    """The registered Qwen3-VL decoder.

    It takes the image rows the ``VISION`` module encodes and each token's
    3-axis M-RoPE position.
    """

    graph_name = "qwen3vl_moe_language"
    consumes_vision: ClassVar[bool] = True
    # The graph keeps the graph-API build's input order.
    input_layout = (
        "tokens",
        "return_n_logits",
        "input_row_offsets",
        "vision",
        "token_positions",
        "signals",
        "kv",
    )

    def __init__(self, config: Qwen3VLConfig) -> None:
        super().__init__()
        self.config = config
        self.kv_params = config.kv_params
        with default_dtype(config.compute_dtype):
            self.model = Qwen3VLTextModel(config)
        self.model.name_transparent = True

    @classmethod
    def token_position_type(
        cls, config: Qwen3VLConfig, device_ref: DeviceRef
    ) -> TensorType:
        """The ``[3, total_seq_len]`` M-RoPE positions, on the host."""
        del config, device_ref
        return TensorType(
            DType.int64, [3, "total_seq_len"], device=DeviceRef.CPU()
        )

    def forward(self, inputs: DecoderInputs[Tensor]) -> DecoderOutputs:
        """Runs the decoder over the batch."""
        assert inputs.vision is not None
        assert inputs.token_positions is not None
        mesh = self.model.norm.weight.mesh
        tokens = _replicate(inputs.tokens, mesh)
        kv_collection = PagedCacheValues.from_upstream(
            tree.leaves(
                self.kv_params.unflatten_kv_inputs(
                    iter(kv_input_values(tree.leaves(inputs.kv)))
                ),
                leaf=KVCacheInputsPerDevice,
            ),
            tokens.mapping,
        )
        return self.model(
            tokens,
            _replicate(inputs.input_row_offsets, mesh),
            inputs.return_n_logits,
            kv_collection,
            inputs.vision,
            inputs.token_positions,
        )
