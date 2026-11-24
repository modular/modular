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
# Ported to ModuleV3 from
# max/python/max/pipelines/architectures/qwen3_5/qwen3_5.py
# at main 4df8111afba.
"""The Qwen3.5 hybrid decoder: Gated DeltaNet and full attention layers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar

from max import functional as F
from max import tree
from max.driver import CPU
from max.dtype import DType
from max.graph import BufferValue, DeviceRef, TensorType, TensorValue
from max.nn import Module, as_subgraph
from max.nn.common_layers.embedding import VocabParallelEmbedding
from max.nn.common_layers.kv_cache import (
    PagedCacheValues,
    cache_tensor,
    kv_input_values,
)
from max.nn.common_layers.mesh_axis import TP
from max.nn.common_layers.quant_linear import (
    QuantizedLinear,
    QuantizedMLP,
    tensor_parallel_mlp,
)
from max.nn.kv_cache import (
    KVCacheInputsPerDevice,
    RecurrentLeafInputs,
    RecurrentStateInputsPerDevice,
)
from max.nn.outputs import ReturnHiddenStates, ReturnLogits
from max.nn.quant_config import QuantConfig
from max.nn.sequential import ModuleList
from max.pipelines.lib import ModuleV3Outputs
from max.pipelines.lib.role_contracts import (
    DecoderInputs,
    DecoderOutputs,
    VisionInputs,
)
from max.pipelines.lib.vlm_utils import F_merge_multimodal_embeddings
from max.tensor import Tensor, default_dtype
from max.tensor.sharding import DeviceMapping, DeviceMesh, NamedMapping

from .layers.attention import Qwen3_5Attention
from .layers.gated_deltanet import (
    GatedDeltaNet,
    GatedDeltaStateAccess,
)
from .layers.norm import RMSNorm
from .layers.rotary import mrope_freqs_cis, rope_freqs_cis
from .model_config import Qwen3_5Config
from .state_cache import ATTN_CACHE_KEY, STATE_CACHE_KEY

__all__ = [
    "FULL_ATTENTION",
    "Qwen3_5Decoder",
    "Qwen3_5DecoderLayer",
    "Qwen3_5TextModel",
    "distributed_state",
    "replicate",
]

FULL_ATTENTION = "full_attention"
LINEAR_ATTENTION = "linear_attention"


def replicate(value: Tensor, mesh: DeviceMesh) -> Tensor:
    """Places a single-device tensor on every device of ``mesh``.

    A broadcast collective, not a peer copy, which device graph capture
    cannot record. Onto one device it is a placement.
    """
    if mesh.num_devices > 1:
        return F.distributed_broadcast(value, mesh)
    return value.to(mesh)


def _mlp(
    hidden_size: int,
    intermediate_size: int,
    *,
    dtype: DType,
    quant_config: QuantConfig | None,
) -> QuantizedMLP:
    """Builds the SwiGLU feed-forward block, split by its intermediate dim.

    Under tensor parallelism it returns the down projection's partial sums.

    Raises:
        ValueError: If the checkpoint's gate and up projections keep scales
            that one fused matmul can't share.
    """
    if quant_config is not None and not quant_config.can_use_fused_mlp:
        raise ValueError(
            "Qwen3.5 fuses each MLP's gate and up projections into one"
            " matmul, so their per-tensor NVFP4 scales must be equal"
        )
    with default_dtype(dtype):
        mlp = QuantizedMLP(
            hidden_size, intermediate_size, quant_config=quant_config
        )
    return tensor_parallel_mlp(mlp)


class Qwen3_5DecoderLayer(Module[..., Tensor]):
    """A pre-norm block: a full or a linear attention mixer, then the MLP."""

    def __init__(
        self,
        config: Qwen3_5Config,
        layer_idx: int,
        rotary_dim: int,
        *,
        kind: str | None = None,
        quantized: bool = True,
    ) -> None:
        """Builds layer ``layer_idx`` of ``config``'s schedule.

        Args:
            config: The model config.
            layer_idx: The layer's index in the model.
            rotary_dim: How many lanes of each attention head rotate.
            kind: The mixer, by default the schedule's for the layer.
            quantized: Whether the checkpoint's quantization scheme covers
                the layer. The MTP head's layer is unquantized.
        """
        compute_dtype = config.compute_dtype
        scheme = config.quant_scheme if quantized else None
        attn_quant = scheme.attn_config(layer_idx) if scheme else None
        mlp_quant = scheme.mlp_config(layer_idx) if scheme else None
        norm_dtype = config.norm_dtype or DType.float32
        eps = config.rms_norm_eps or 1e-6
        self.kind = kind or config.layer_types[layer_idx]
        if self.kind == FULL_ATTENTION:
            self.self_attn: Qwen3_5Attention | None = Qwen3_5Attention(
                hidden_size=config.hidden_size,
                num_attention_heads=config.num_attention_heads,
                num_key_value_heads=config.num_key_value_heads,
                head_dim=config.attn_kv_params.head_dim,
                rotary_dim=rotary_dim,
                kv_params=config.attn_kv_params,
                layer_idx=layer_idx,
                dtype=compute_dtype,
                norm_dtype=config.norm_dtype or compute_dtype,
                norm_eps=eps,
                scale=config.attention_multiplier,
                quant_config=attn_quant,
            )
            self.linear_attn: GatedDeltaNet | None = None
        else:
            self.self_attn = None
            self.linear_attn = GatedDeltaNet(
                hidden_size=config.hidden_size,
                num_key_heads=config.linear_num_key_heads,
                num_value_heads=config.linear_num_value_heads,
                key_head_dim=config.linear_key_head_dim,
                value_head_dim=config.linear_value_head_dim,
                conv_kernel_size=config.linear_conv_kernel_dim,
                dtype=compute_dtype,
                rms_norm_eps=eps,
                ssm_dtype=config.mamba_ssm_dtype,
                quant_config=attn_quant,
                scalar_dtypes=config.linear_attn_scalar_dtypes,
            )
        self.mlp = _mlp(
            config.hidden_size,
            config.intermediate_size,
            dtype=compute_dtype,
            quant_config=mlp_quant,
        )
        self.input_layernorm = RMSNorm(
            config.hidden_size, eps=eps, dtype=norm_dtype
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=eps, dtype=norm_dtype
        )

    def forward(self, h: Tensor, *mixer_args: Any) -> Tensor:
        """Runs the block; the mixer reads ``mixer_args`` after its input."""
        mixer = (
            self.self_attn if self.self_attn is not None else self.linear_attn
        )
        assert mixer is not None
        # The row-parallel projections leave partial sums, reduced before
        # each residual add.
        h = h + mixer(self.input_layernorm(h), *mixer_args).to(h.mapping)
        return h + self.mlp(self.post_attention_layernorm(h)).to(h.mapping)


def distributed_state(
    per_device: Sequence[
        RecurrentStateInputsPerDevice[TensorValue, BufferValue]
    ],
    mapping: DeviceMapping,
) -> RecurrentStateInputsPerDevice[Tensor, Tensor]:
    """Builds one state tensor per leaf from each device's pool.

    A pool holds its own device's heads, so it is split on its head axis;
    a request's rows are the same on every device, so they take the
    replicated ``mapping``.
    """
    leaves: list[RecurrentLeafInputs[Tensor, Tensor]] = []
    for index in range(len(per_device[0].leaves)):
        pools = [device.leaves[index].pool for device in per_device]
        spec = (None, TP) + (None,) * (pools[0].rank - 2)
        leaves.append(
            RecurrentLeafInputs(
                pool=cache_tensor(pools, NamedMapping(mapping.mesh, spec)),
                live_row_ids=cache_tensor(
                    [
                        device.leaves[index].live_row_ids
                        for device in per_device
                    ],
                    mapping,
                ),
            )
        )
    return RecurrentStateInputsPerDevice(leaves=tuple(leaves))


class Qwen3_5TextModel(Module[..., ModuleV3Outputs]):
    """The embedding, the hybrid layer stack, the final norm and the head.

    Every Nth layer is full attention, with a KV cache; the rest are Gated
    DeltaNet, with a recurrent state.
    """

    def __init__(self, config: Qwen3_5Config) -> None:
        if config.model_quantization_encoding is not None:
            raise NotImplementedError(
                "GGUF and GPTQ Qwen3.5 checkpoints are not supported"
            )
        self.config = config
        compute_dtype = config.compute_dtype
        self.rotary_dim = int(
            config.attn_kv_params.head_dim * config.partial_rotary_factor
        )
        self.mrope_enabled = config.mrope_enabled
        self.layer_types = list(config.layer_types)
        self.linear_layer_indices = [
            i
            for i, kind in enumerate(self.layer_types)
            if kind != FULL_ATTENTION
        ]
        self.use_subgraphs = config.use_subgraphs
        with default_dtype(compute_dtype):
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size, dim=config.hidden_size
            )
        self.layers = ModuleList(
            [
                Qwen3_5DecoderLayer(config, i, self.rotary_dim)
                for i in range(len(self.layer_types))
            ]
        )
        self.norm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps or 1e-6,
            dtype=config.norm_dtype or DType.float32,
        )
        scheme = config.quant_scheme
        self.tie_word_embeddings = config.tie_word_embeddings
        self.lm_head: QuantizedLinear | None = None
        if not config.tie_word_embeddings:
            with default_dtype(compute_dtype):
                lm_head = QuantizedLinear(
                    config.hidden_size,
                    config.vocab_size,
                    bias=False,
                    quant_config=(
                        scheme.mlp
                        if scheme and scheme.quantize_lm_head
                        else None
                    ),
                )
            # Column-parallel: each device projects onto its share of the
            # vocabulary.
            self.lm_head = lm_head.to(
                NamedMapping(lm_head.weight.mesh, (TP, None))
            )

    @property
    def mesh(self) -> DeviceMesh:
        """The devices the model runs on."""
        return self.norm.weight.mesh

    def static_freqs_cis(self) -> Tensor:
        """The rotary table by position, on the host."""
        config = self.config
        return rope_freqs_cis(
            config.rope_theta, self.rotary_dim, config.max_seq_len
        )

    def mrope_freqs_cis(self, position_ids: Tensor) -> Tensor:
        """The rotary table by token, from ``[3, total_seq_len]`` positions."""
        config = self.config
        assert config.mrope_section is not None
        return mrope_freqs_cis(
            position_ids,
            theta=config.rope_theta,
            rotary_dim=self.rotary_dim,
            mrope_section=config.mrope_section,
        )

    def hidden_states(
        self,
        tokens: Tensor,
        input_row_offsets: Tensor,
        kv_collection: PagedCacheValues,
        state: RecurrentStateInputsPerDevice[Tensor, Tensor] | None,
        *,
        vision: VisionInputs[Tensor] | None = None,
        position_ids: Tensor | None = None,
        capture_layers: Sequence[int] = (),
        subgraphs: bool = True,
    ) -> tuple[Tensor, list[Tensor]]:
        """Returns every token's last hidden state, before the final norm.

        Args:
            tokens: The ragged token ids, replicated on the mesh.
            input_row_offsets: The ragged row offsets, replicated.
            kv_collection: The full-attention layers' cache.
            state: The recurrent-state leaves: the conv window, the
                recurrent state and, in a verify, the ring. ``None`` when
                the model has no linear layers.
            vision: The batch's image rows and their token positions.
            position_ids: ``[3, total_seq_len]`` M-RoPE positions, replicated.
                ``None`` keeps the static table, indexed by
                ``cache_length + token_idx``.
            capture_layers: The layers whose output to return as well.
            subgraphs: Whether layers of one kind share a subgraph. A
                speculative verify turns it off: what each linear layer
                captures for the rollback can't leave a subgraph.

        Returns:
            The hidden states, and the captured layers' outputs in order.
        """
        mesh = tokens.mesh
        h = self.embed_tokens(tokens)
        if vision is not None:
            h = F_merge_multimodal_embeddings(
                h, vision.embeddings, vision.scatter_indices
            )
        replicated = DeviceMapping.replicated(mesh)
        freq_row_ids: Tensor | None = None
        if position_ids is not None:
            # The table is per token: row `i` is token `i`'s, so each token
            # reads its own row index rather than its cache position.
            freqs_cis = self.mrope_freqs_cis(position_ids)
            freq_row_ids = F.range(
                0,
                h.shape[0],
                1,
                out_dim=h.shape[0],
                dtype=DType.uint32,
                device=mesh,
            ).unsqueeze(0)
        else:
            freqs_cis = self.static_freqs_cis().to(replicated)

        # The conv window, the recurrent state and, in a verify, the ring.
        leaves = state.leaves if state is not None else ()

        captures: list[Tensor] = []
        capture_set = set(capture_layers)
        attention_index = 0
        linear_index = 0
        for index, layer in enumerate(self.layers):
            kind = self.layer_types[index]
            mixer_args: tuple[object, ...]
            if kind == FULL_ATTENTION:
                # Only the full-attention layers hold KV cache layers, so
                # model layer 3 is cache layer 0.
                mixer_args = (
                    kv_collection,
                    F.constant(attention_index, DType.uint32, device=CPU()),
                    freqs_cis,
                    input_row_offsets,
                    freq_row_ids,
                )
                attention_index += 1
            else:
                # The layers share one subgraph, so each call selects its
                # own rows here.
                conv, recurrent = leaves[:2]
                ring = leaves[2] if len(leaves) > 2 else None
                mixer_args = (
                    GatedDeltaStateAccess(
                        conv_pool=conv.pool,
                        conv_row_id=conv.live_row_ids[linear_index],
                        recurrent_pool=recurrent.pool,
                        recurrent_row_id=recurrent.live_row_ids[linear_index],
                        ring_pool=None if ring is None else ring.pool,
                        ring_row_id=(
                            None
                            if ring is None
                            else ring.live_row_ids[linear_index]
                        ),
                    ),
                    input_row_offsets,
                )
                linear_index += 1
            call = (
                as_subgraph(layer, name=f"qwen3_5_{kind}_block")
                if subgraphs and self.use_subgraphs
                else layer
            )
            h = call(h, *mixer_args)
            if index in capture_set:
                captures.append(h)
        return h, captures

    def head(self, x: Tensor) -> Tensor:
        """Returns the head's logits of ``x``, gathered on the first device.

        Each device projects onto its share of the vocabulary.
        """
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
        """Returns the rows of each sequence's last ``count`` tokens.

        Args:
            input_row_offsets: The ragged row offsets.
            count: The host scalar count of rows per sequence.
        """
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
        state: RecurrentStateInputsPerDevice[Tensor, Tensor] | None,
        vision: VisionInputs[Tensor] | None = None,
        position_ids: Tensor | None = None,
    ) -> ModuleV3Outputs:
        """Runs the decoder and returns the logits it was built to."""
        config = self.config
        capture = (
            list(config.target_layer_ids or ())
            if config.return_hidden_states == ReturnHiddenStates.SELECTED_LAYERS
            else []
        )
        h, captures = self.hidden_states(
            tokens,
            input_row_offsets,
            kv_collection,
            state,
            vision=vision,
            position_ids=position_ids,
            capture_layers=capture,
        )
        last_rows = input_row_offsets[1:] - 1
        last = self.logits(F.gather(h, last_rows, axis=0))
        logits: Tensor | None = None
        offsets: Tensor | None = None
        if config.return_logits == ReturnLogits.ALL:
            logits = self.logits(h)
            offsets = input_row_offsets.local_shards[0]
        elif config.return_logits == ReturnLogits.VARIABLE:
            count = return_n_logits[0]
            rows = self.window_rows(input_row_offsets, count)
            logits = self.logits(F.gather(h, rows, axis=0))
            offsets = F.range(
                0,
                rows.shape[0] + count,
                count,
                out_dim="logit_offsets",
                dtype=DType.int64,
                device=tokens.local_shards[0].device,
            )
        hidden: Tensor | None = None
        mode = config.return_hidden_states
        if mode == ReturnHiddenStates.LAST:
            hidden = F.gather(h, last_rows, axis=0)
        elif mode == ReturnHiddenStates.ALL:
            hidden = h
        elif mode == ReturnHiddenStates.ALL_NORMALIZED:
            hidden = self.norm(h)
        elif mode == ReturnHiddenStates.SELECTED_LAYERS:
            hidden = F.concat(captures, axis=-1)
        return ModuleV3Outputs(
            next_token_logits=last,
            logits=logits,
            logit_offsets=offsets,
            hidden_states=None if hidden is None else hidden.local_shards[0],
        )


class Qwen3_5Decoder(Module[[DecoderInputs[Tensor]], DecoderOutputs]):
    """The registered Qwen3.5 decoder.

    It takes the image rows the ``VISION`` module encodes and, on a checkpoint
    with a vision tower, each token's 3-axis M-RoPE position.
    """

    graph_name = "qwen3_5"
    consumes_vision: ClassVar[bool] = True
    checkpoints_recurrent_state: ClassVar[bool] = True
    # The graph keeps the graph-API build's input order, which Mach binds.
    input_layout = (
        "tokens",
        "input_row_offsets",
        "return_n_logits",
        "signals",
        "kv",
        "vision",
        "token_positions",
    )

    def __init__(self, config: Qwen3_5Config) -> None:
        super().__init__()
        self.config = config
        self.kv_params = config.kv_params
        self.model = Qwen3_5TextModel(config)
        self.model.name_transparent = True

    @classmethod
    def token_position_type(
        cls, config: Qwen3_5Config, device_ref: DeviceRef
    ) -> TensorType | None:
        """The ``[3, total_seq_len]`` M-RoPE positions, when the model has them.

        A checkpoint without a vision tower keeps the static table.
        """
        if not config.mrope_enabled:
            return None
        return TensorType(DType.int64, [3, "total_seq_len"], device=device_ref)

    def forward(self, inputs: DecoderInputs[Tensor]) -> DecoderOutputs:
        """Runs the decoder over the batch."""
        mesh = self.model.mesh
        tokens = replicate(inputs.tokens, mesh)
        input_row_offsets = replicate(inputs.input_row_offsets, mesh)
        kv_tree = self.kv_params.unflatten_kv_inputs(
            iter(kv_input_values(tree.leaves(inputs.kv)))
        )
        state: RecurrentStateInputsPerDevice[Tensor, Tensor] | None = None
        if isinstance(kv_tree, dict):
            attn = kv_tree[ATTN_CACHE_KEY]
            state = distributed_state(
                tree.leaves(
                    kv_tree[STATE_CACHE_KEY],
                    leaf=RecurrentStateInputsPerDevice,
                ),
                tokens.mapping,
            )
        else:
            attn = kv_tree
        kv_collection = PagedCacheValues.from_upstream(
            tree.leaves(attn, leaf=KVCacheInputsPerDevice), tokens.mapping
        )
        position_ids = (
            None
            if inputs.token_positions is None
            else replicate(inputs.token_positions, mesh)
        )
        return self.model(
            tokens,
            input_row_offsets,
            inputs.return_n_logits,
            kv_collection,
            state,
            inputs.vision,
            position_ids,
        )
