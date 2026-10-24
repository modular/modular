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
"""Qwen3.5 hybrid attention model (linear + full attention layers)."""

from __future__ import annotations

import functools
from collections import OrderedDict
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

from max import tree
from max.dtype import DType
from max.graph import (
    BufferType,
    BufferValue,
    DeviceRef,
    Graph,
    ShardingStrategy,
    TensorType,
    TensorValue,
    TensorValueLike,
    Type,
    Value,
    ops,
)
from max.graph import Module as GraphModule
from max.graph.quantization import QuantizationEncoding
from max.nn.comm import Allreduce
from max.nn.embedding import VocabParallelEmbedding
from max.nn.graph_groups import MRoPEPositions
from max.nn.kv_cache import (
    KVCacheInputsPerDevice,
    KVCacheParamInterface,
    PagedCacheValues,
    RecurrentStateInputsPerDevice,
    packed_page_stride,
)
from max.nn.layer import LayerList, Module
from max.nn.linear import MLP, ColumnParallelLinear, Linear
from max.nn.norm import RMSNorm
from max.nn.quant_config import QuantConfig
from max.nn.rotary_embedding import Llama3RotaryEmbedding
from max.nn.state_space import GatedDeltaReplayInputs
from max.nn.transformer import (
    DecoderOutputs,
    ReturnHiddenStates,
    forward_sequential_layers,
)
from max.nn.transformer.distributed_transformer import (
    DistributedLogitsPostprocessMixin,
)
from max.nn.transformer.transformer import forward_sharded_layers
from max.pipelines.lib.interfaces.batch_processor import (
    ragged_batch_inputs,
    signal_inputs,
    vision_merge_inputs,
)
from max.pipelines.lib.vlm_utils import merge_multimodal_embeddings
from max.tree import Tree

from .layers.attention import Qwen3_5Attention
from .layers.gated_deltanet import GatedDeltaNet
from .layers.text_rotary import Qwen3_5TextRotaryEmbedding
from .layers.visual_transformer import Qwen3_5VisionTransformer
from .model_config import Qwen3_5Config
from .quantization import storage_dtype
from .state_cache import (
    ATTN_CACHE_KEY,
    STATE_CACHE_KEY,
    GatedDeltaStateAccess,
    layer_state_access,
)


def _shard_mlp_and_norms(
    mlp: MLP,
    input_layernorm: RMSNorm,
    post_attention_layernorm: RMSNorm,
    devices: list[DeviceRef],
) -> tuple[Sequence[MLP], Sequence[RMSNorm], Sequence[RMSNorm]]:
    """Splits the MLP by intermediate dim and replicates the two norms.

    Shared by both block kinds, which differ only in their mixer. Sharding is
    unconditional: at one device every strategy is the identity, which is what
    every other distributed architecture in the tree does.

    Args:
        mlp: The block's feed-forward network.
        input_layernorm: The norm before the mixer.
        post_attention_layernorm: The norm before the MLP.
        devices: Devices to shard across.

    Returns:
        The per-device MLP, input-norm and post-attention-norm shards.
    """
    num_devices = len(devices)
    replicate = ShardingStrategy.replicate(num_devices)

    mlp.sharding_strategy = ShardingStrategy.tensor_parallel(num_devices)
    input_layernorm.sharding_strategy = replicate
    post_attention_layernorm.sharding_strategy = replicate
    return (
        mlp.shard(devices),
        input_layernorm.shard(devices),
        post_attention_layernorm.shard(devices),
    )


class Qwen3_5FullAttentionBlock(Module):
    """Full-attention transformer block (KV cache path)."""

    def __init__(
        self,
        config: Qwen3_5Config,
        layer_idx: int,
        rope: Llama3RotaryEmbedding,
        create_norm: Callable[..., RMSNorm],
        linear_cls: Callable[..., Linear],
        attn_quant_config: QuantConfig | None = None,
        mlp_quant_config: QuantConfig | None = None,
    ) -> None:
        super().__init__()
        compute_dtype = config.compute_dtype
        self.self_attn = Qwen3_5Attention(
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            hidden_size=config.hidden_size,
            head_dim=config.attn_kv_params.head_dim,
            kv_params=config.attn_kv_params,
            layer_idx=layer_idx,
            dtype=storage_dtype(attn_quant_config, compute_dtype),
            rope=rope,
            linear_cls=linear_cls,
            devices=config.devices,
            scale=config.attention_multiplier,
            partial_rotary_factor=config.partial_rotary_factor,
            has_bias=config.attention_bias,
            norm_dtype=config.norm_dtype or compute_dtype,
            norm_eps=config.rms_norm_eps or 1e-6,
            quant_config=attn_quant_config,
        )
        self.mlp = MLP(
            storage_dtype(mlp_quant_config, compute_dtype),
            config.model_quantization_encoding,
            config.hidden_size,
            config.intermediate_size,
            config.devices,
            linear_cls,
            quant_config=mlp_quant_config,
        )
        self.input_layernorm = create_norm()
        self.post_attention_layernorm = create_norm()

        self.self_attn.sharding_strategy = ShardingStrategy.tensor_parallel(
            len(config.devices)
        )
        self.self_attn_shards = self.self_attn.shard(config.devices)
        (
            self.mlp_shards,
            self.input_layernorm_shards,
            self.post_attention_layernorm_shards,
        ) = _shard_mlp_and_norms(
            self.mlp,
            self.input_layernorm,
            self.post_attention_layernorm,
            config.devices,
        )
        self.allreduce = Allreduce(num_accelerators=len(config.devices))

    def __call__(
        self,
        xs: list[TensorValue],
        layer_idx: TensorValue,
        signal_buffers: list[BufferValue],
        kv_blocks: list[BufferValue],
        cache_lengths: list[TensorValue],
        lookup_table: list[TensorValue],
        max_prompt_length: list[TensorValue],
        max_cache_length: list[TensorValue],
        attention_dispatch_metadata: list[TensorValue],
        freqs_cis: list[TensorValue],
        input_row_offsets: list[TensorValue],
        freq_row_ids: list[TensorValue] | None = None,
    ) -> list[TensorValue]:
        norm_xs = forward_sharded_layers(self.input_layernorm_shards, xs)
        # `o_proj` is row-parallel, so each device holds a partial sum.
        attn_outs = self.allreduce(
            [
                shard(
                    layer_idx,
                    norm_xs[i],
                    PagedCacheValues(
                        kv_blocks=kv_blocks[i],
                        page_stride=packed_page_stride(kv_blocks[i]),
                        cache_lengths=cache_lengths[i],
                        lookup_table=lookup_table[i],
                        max_prompt_length=max_prompt_length[i],
                        max_cache_length=max_cache_length[i],
                        attention_dispatch_metadata=(
                            attention_dispatch_metadata[i]
                        ),
                    ),
                    freqs_cis[i],
                    input_row_offsets[i],
                    None if freq_row_ids is None else freq_row_ids[i],
                )
                for i, shard in enumerate(self.self_attn_shards)
            ],
            signal_buffers,
        )
        hs = [x + attn_out for x, attn_out in zip(xs, attn_outs, strict=True)]
        norm_hs = forward_sharded_layers(
            self.post_attention_layernorm_shards, hs
        )
        mlp_outs = self.allreduce(
            forward_sharded_layers(self.mlp_shards, norm_hs), signal_buffers
        )
        return [h + mlp_out for h, mlp_out in zip(hs, mlp_outs, strict=True)]


class Qwen3_5LinearAttentionBlock(Module):
    """Linear-attention transformer block (Gated DeltaNet path)."""

    replay_capture: list[list[GatedDeltaReplayInputs]] | None = None
    """Per-device sink for this layer's state-kernel inputs, or ``None``.

    Set by the speculative graph, which re-runs those kernels over the
    accepted prefix after the verify pass. Off by default, so the base graph
    is unchanged.
    """

    verify_width: TensorValue | None = None
    """The speculative graph's verify width operand, or ``None``.

    Set alongside :attr:`replay_capture`. See :class:`GatedDeltaNet`.
    """

    def __init__(
        self,
        config: Qwen3_5Config,
        create_norm: Callable[..., RMSNorm],
        linear_cls: Callable[..., Linear],
        attn_quant_config: QuantConfig | None = None,
        mlp_quant_config: QuantConfig | None = None,
    ) -> None:
        super().__init__()
        compute_dtype = config.compute_dtype
        self.linear_attn = GatedDeltaNet(
            hidden_size=config.hidden_size,
            num_key_heads=config.linear_num_key_heads,
            num_value_heads=config.linear_num_value_heads,
            key_head_dim=config.linear_key_head_dim,
            value_head_dim=config.linear_value_head_dim,
            conv_kernel_size=config.linear_conv_kernel_dim,
            dtype=compute_dtype,
            device=config.devices[0],
            rms_norm_eps=config.rms_norm_eps or 1e-6,
            ssm_dtype=config.mamba_ssm_dtype,
            proj_dtype=storage_dtype(attn_quant_config, compute_dtype),
            quant_config=attn_quant_config,
        )
        self.mlp = MLP(
            storage_dtype(mlp_quant_config, compute_dtype),
            config.model_quantization_encoding,
            config.hidden_size,
            config.intermediate_size,
            config.devices,
            linear_cls,
            quant_config=mlp_quant_config,
        )
        self.input_layernorm = create_norm()
        self.post_attention_layernorm = create_norm()

        self.linear_attn.sharding_strategy = ShardingStrategy.tensor_parallel(
            len(config.devices)
        )
        self.linear_attn_shards = self.linear_attn.shard(config.devices)
        (
            self.mlp_shards,
            self.input_layernorm_shards,
            self.post_attention_layernorm_shards,
        ) = _shard_mlp_and_norms(
            self.mlp,
            self.input_layernorm,
            self.post_attention_layernorm,
            config.devices,
        )
        self.allreduce = Allreduce(num_accelerators=len(config.devices))

    def __call__(
        self,
        xs: list[TensorValue],
        signal_buffers: list[BufferValue],
        state: list[GatedDeltaStateAccess],
        input_row_offsets: list[TensorValue],
    ) -> list[TensorValue]:
        norm_xs = forward_sharded_layers(self.input_layernorm_shards, xs)
        # Each device owns a slice of the value heads, so `out_proj` emits a
        # partial sum over the full hidden dim. The state rows arrive already
        # selected for this layer (see ``layer_state_access``).
        attn_outs = self.allreduce(
            [
                shard(
                    norm_xs[i],
                    conv_pool=state[i].conv_pool,
                    conv_row_id=state[i].conv_row_id,
                    recurrent_pool=state[i].recurrent_pool,
                    recurrent_row_id=state[i].recurrent_row_id,
                    input_row_offsets=input_row_offsets[i],
                    replay_capture=(
                        None
                        if self.replay_capture is None
                        else self.replay_capture[i]
                    ),
                    ring=state[i].ring,
                    verify_width=self.verify_width,
                )
                for i, shard in enumerate(self.linear_attn_shards)
            ],
            signal_buffers,
        )
        hs = [x + attn_out for x, attn_out in zip(xs, attn_outs, strict=True)]
        norm_hs = forward_sharded_layers(
            self.post_attention_layernorm_shards, hs
        )
        mlp_outs = self.allreduce(
            forward_sharded_layers(self.mlp_shards, norm_hs), signal_buffers
        )
        return [h + mlp_out for h, mlp_out in zip(hs, mlp_outs, strict=True)]


def rotary_embedding(
    config: Qwen3_5Config, *, mrope: bool
) -> Llama3RotaryEmbedding:
    """Returns the full-attention layers' rotary embedding, which rotates
    only the partial rotary dims, over 3-axis positions under ``mrope``."""
    rotary_dim = int(
        config.attn_kv_params.head_dim * config.partial_rotary_factor
    )
    if mrope:
        assert config.mrope_section is not None
        return Qwen3_5TextRotaryEmbedding(
            dim=config.hidden_size,
            n_heads=config.num_attention_heads,
            theta=config.rope_theta,
            max_seq_len=config.max_seq_len,
            mrope_section=config.mrope_section,
            head_dim=rotary_dim,
            interleaved=config.interleaved_rope_weights,
            scaling_params=config.rope_scaling_params,
        )
    return Llama3RotaryEmbedding(
        dim=config.hidden_size,
        n_heads=config.num_attention_heads,
        theta=config.rope_theta,
        max_seq_len=config.max_seq_len,
        head_dim=rotary_dim,
        interleaved=config.interleaved_rope_weights,
        scaling_params=config.rope_scaling_params,
    )


def norm_factory(config: Qwen3_5Config) -> Callable[..., RMSNorm]:
    """Returns the factory of Qwen3.5's RMSNorms, which scale by
    ``1 + weight``.

    Raises:
        ValueError: If the config doesn't name RMSNorm and its epsilon.
    """
    if config.norm_method != "rms_norm" or config.rms_norm_eps is None:
        raise ValueError(
            "Qwen3.5 requires RMSNorm. Set norm_method='rms_norm' "
            "and provide rms_norm_eps."
        )
    return functools.partial(
        RMSNorm,
        config.hidden_size,
        dtype=config.norm_dtype or DType.float32,
        eps=config.rms_norm_eps,
        weight_offset=1.0,
        multiply_before_cast=False,
    )


class Qwen3_5(DistributedLogitsPostprocessMixin, Module):
    """Qwen3.5 hybrid attention model.

    This model uses a mix of full attention (with KV cache) and linear
    attention (Gated DeltaNet) layers. Every full_attention_interval-th
    layer uses full attention, and the rest use linear attention.
    """

    def __init__(self, config: Qwen3_5Config) -> None:
        super().__init__()
        self.config = config
        self.devices = config.devices
        self.num_devices = len(config.devices)

        if config.model_quantization_encoding == QuantizationEncoding.GPTQ:
            raise NotImplementedError("GPTQ Qwen3.5 is not implemented yet")
        if config.model_quantization_encoding is not None:
            raise NotImplementedError("GGUFQ Qwen3.5 is not implemented yet")

        # An image compresses many soft-token patches into far fewer
        # position steps on each of three axes, so every token after it needs
        # an explicit 3-axis position rather than the kernels' default
        # `cache_length + token_idx`. Text-only checkpoints have no image to
        # splice and degenerate to that default on all three axes, so they
        # keep the static table.
        #
        # The KV cache dtype does not enter into this: rope is applied to K
        # before the value is cast into the cache.
        self.mrope_enabled = config.mrope_enabled
        rope = rotary_embedding(config, mrope=self.mrope_enabled)
        self.rope = rope
        create_norm = norm_factory(config)

        # Quantization is per module, not per model: this checkpoint's MLPs are
        # NVFP4 while its attention and GDN projections are per-tensor FP8, so
        # each construction site is handed its own config rather than one
        # partial-bound `linear_cls`.
        scheme = config.quant_scheme
        compute_dtype = config.compute_dtype
        linear_cls = Linear

        self.layer_types = config.layer_types
        self.linear_layer_indices = [
            i
            for i, lt in enumerate(config.layer_types)
            if lt == "linear_attention"
        ]

        layers: list[Module] = []
        for i, lt in enumerate(config.layer_types):
            attn_quant_config = scheme.attn_config(i) if scheme else None
            mlp_quant_config = scheme.mlp_config(i) if scheme else None
            if lt == "full_attention":
                layers.append(
                    Qwen3_5FullAttentionBlock(
                        config=config,
                        layer_idx=i,
                        rope=rope,
                        create_norm=create_norm,
                        linear_cls=linear_cls,
                        attn_quant_config=attn_quant_config,
                        mlp_quant_config=mlp_quant_config,
                    )
                )
            else:
                layers.append(
                    Qwen3_5LinearAttentionBlock(
                        config=config,
                        create_norm=create_norm,
                        linear_cls=linear_cls,
                        attn_quant_config=attn_quant_config,
                        mlp_quant_config=mlp_quant_config,
                    )
                )
        self.layers = LayerList(layers)

        # Final norm (replicated across devices)
        self.norm = create_norm()
        self.norm.sharding_strategy = ShardingStrategy.replicate(
            self.num_devices
        )
        self.norm_shards = self.norm.shard(config.devices)

        # Embedding and output layers. `embed_tokens` is never quantized;
        # `lm_head` is NVFP4 in this checkpoint, and shares the MLP's config.
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            compute_dtype,
            config.devices,
        )
        lm_head_quant_config = (
            scheme.mlp if scheme and scheme.quantize_lm_head else None
        )
        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            storage_dtype(lm_head_quant_config, compute_dtype),
            devices=config.devices,
            tied_weight=(
                self.embed_tokens.weight if config.tie_word_embeddings else None
            ),
            quant_config=lm_head_quant_config,
        )

        self.kv_params = config.kv_params
        self.return_logits = config.return_logits
        self.return_hidden_states = config.return_hidden_states

        # Vision encoder (only present in multimodal checkpoints)
        self.vision_encoder: Qwen3_5VisionTransformer | None = (
            Qwen3_5VisionTransformer(config=config.vision_config)
            if config.vision_config is not None
            else None
        )

    def __call__(
        self,
        tokens: TensorValueLike,
        kv_collections: list[PagedCacheValues],
        return_n_logits: TensorValue,
        input_row_offsets: TensorValue,
        signal_buffers: list[BufferValue],
        state: list[RecurrentStateInputsPerDevice[TensorValue, BufferValue]],
        image_embeddings: list[TensorValue] | None = None,
        image_token_indices: list[TensorValue] | None = None,
        position_ids: TensorValue | None = None,
    ) -> DecoderOutputs:
        """Forward pass through the hybrid model.

        The state pools are mutable graph inputs the SSM kernels write in
        place; the only graph outputs are the logits.

        Under tensor parallelism the hidden state is replicated on every
        device while the heads are split, so each device's pools hold its own
        share of the heads.

        Args:
            tokens: Input token IDs.
            kv_collections: KV cache per device.
            return_n_logits: Number of logits to return.
            input_row_offsets: Row offsets for ragged batching.
            signal_buffers: Signal buffers for allreduce.
            state: Per-device recurrent-state inputs: each leaf's pool and
                the rows this step reads and writes.
            image_embeddings: Per-device vision encoder output to merge into
                token embeddings. Shape [vision_merged_seq_len, hidden_size].
                None for text-only.
            image_token_indices: Per-device scatter indices for placing image
                embeddings in the token sequence. Shape
                [vision_merged_seq_len]. None for text-only.
            position_ids: ``[3, total_seq_len]`` temporal/height/width M-RoPE
                positions, one column per token of the ragged batch. Required
                when :attr:`mrope_enabled`; ``None`` leaves the rotary on the
                static table indexed by ``cache_length + token_idx``, which is
                what the speculative graph's text-only draft uses.

        Returns:
            Tuple of (logits,).
        """
        hs = self.embed_tokens(tokens, signal_buffers)

        if image_embeddings is not None and image_token_indices is not None:
            # The hidden state is replicated across devices, so each replica
            # merges the same embeddings at the same positions.
            hs = [
                merge_multimodal_embeddings(h, embeddings, indices)
                for h, embeddings, indices in zip(
                    hs, image_embeddings, image_token_indices, strict=True
                )
            ]

        # With M-RoPE the table is per token, not per position: row `i` holds
        # the frequencies for token `i` of the ragged batch. The rope kernels
        # index it by an explicit position id, so they are handed the token's
        # own row index instead of their default cache-derived position.
        freq_row_ids: list[TensorValue] | None = None
        if position_ids is not None:
            assert isinstance(self.rope, Qwen3_5TextRotaryEmbedding)
            # Unlike the static table, this one depends on the step's
            # positions and cannot be hoisted to init. Copying it GPU to GPU
            # joins the per-device capture sequences, which device graph
            # capture rejects, so each device builds its own from a broadcast.
            freqs_cis = [
                self.rope.freqs_cis_position_ids(device_position_ids)
                for device_position_ids in ops.distributed_broadcast(
                    position_ids, signal_buffers
                )
            ]
            freq_row_ids = [
                ops.unsqueeze(
                    ops.range(
                        0,
                        hs[0].shape[0],
                        1,
                        device=device,
                        dtype=DType.uint32,
                    ),
                    0,
                )
                for device in self.devices
            ]
        else:
            freqs_cis = [
                self.rope.freqs_cis.to(device) for device in self.devices
            ]
        row_offsets = ops.distributed_broadcast(
            input_row_offsets.to(self.devices[0]), signal_buffers
        )

        # An unscaled FP8 cache (`--kv-cache-format float8_e4m3fn`) is
        # supported and needs no scale buffers; a *scaled* one is not.
        for kv_collection in kv_collections:
            assert kv_collection.kv_scales is None, (
                "Qwen3.5 does not support a scale-calibrated quantized KV cache"
            )
            assert kv_collection.draft_attention_dispatch_metadata is None, (
                "Qwen3.5 does not support eagle speculation"
            )
            assert kv_collection.attention_dispatch_metadata is not None
        kv_cache_idx = 0

        def inputs_for_layer(
            idx: int, hs: list[TensorValue]
        ) -> list[Tree[Any]]:
            nonlocal kv_cache_idx
            if self.layer_types[idx] == "full_attention":
                # ``layer_idx`` is the sequential index within the KV cache
                # (0-based across full-attention layers only), distinct from
                # the absolute layer index. The KV cache is only allocated for
                # full-attention layers.
                layer_idx_tensor = ops.constant(
                    kv_cache_idx, DType.uint32, device=DeviceRef.CPU()
                )
                kv_cache_idx += 1
                # ``forward_sequential_layers`` only introspects ``Value`` and
                # sequences of them, so the per-device dataclasses are
                # unpacked field by field.
                full_attn_inputs: list[Tree[Any]] = [
                    hs,
                    layer_idx_tensor,
                    signal_buffers,
                    [kv.kv_blocks for kv in kv_collections],
                    [kv.cache_lengths for kv in kv_collections],
                    [kv.lookup_table for kv in kv_collections],
                    [kv.max_prompt_length for kv in kv_collections],
                    [kv.max_cache_length for kv in kv_collections],
                    [
                        kv.attention_dispatch_metadata
                        for kv in kv_collections
                        if kv.attention_dispatch_metadata is not None
                    ],
                    freqs_cis,
                    row_offsets,
                ]
                if freq_row_ids is not None:
                    full_attn_inputs.append(freq_row_ids)
                return full_attn_inputs
            # All linear layers share one compiled subgraph, whose body is
            # built from the first of them. Only the arguments vary per call,
            # so this layer's row of ``live_row_ids`` has to be selected
            # here -- selecting it inside the block would bake layer 0's
            # row into every layer's state access.
            layer = self.linear_layer_indices.index(idx)
            return [
                hs,
                signal_buffers,
                layer_state_access(state, layer),
                row_offsets,
            ]

        full_attn_indices = [
            i for i, lt in enumerate(self.layer_types) if lt == "full_attention"
        ]
        groups: list[list[int]] = [
            g for g in (full_attn_indices, self.linear_layer_indices) if g
        ]

        # DFlash-style drafters read the target's hidden state at a few
        # chosen layers. `on_layer_output` fires for both layer kinds and
        # through the subgraph path, so the taps need no per-kind casing.
        capture_hidden_states: list[list[TensorValue]] | None = None
        on_layer_output = None
        if (
            self.return_hidden_states == ReturnHiddenStates.SELECTED_LAYERS
            and self.config.target_layer_ids
        ):
            capture_layer_set = set(self.config.target_layer_ids)
            captures: list[list[TensorValue]] = []
            capture_hidden_states = captures

            def on_layer_output(idx: int, h: list[TensorValue]) -> None:
                if idx in capture_layer_set:
                    captures.append(list(h))

        hs = forward_sequential_layers(
            list(self.layers),
            inputs_for_layer=inputs_for_layer,
            initial_hidden_states=hs,
            on_layer_output=on_layer_output,
            subgraph_layer_groups=(
                groups if self.config.use_subgraphs else None
            ),
            name_for_subgraph=lambda g: (
                f"qwen3_5_{self.layer_types[groups[g][0]]}_block"
            ),
            weight_prefix_for_layer=lambda i: f"layers.{i}.",
        )

        return self._postprocess_logits(
            hs,
            row_offsets,
            return_n_logits,
            signal_buffers,
            capture_hidden_states=capture_hidden_states,
        )

    def input_tree(
        self, kv_params: KVCacheParamInterface
    ) -> OrderedDict[str, object]:
        """Names the graph's inputs; its leaves are the graph's inputs.

        The recurrent state sits inside ``kv`` beside the attention cache.
        Vision and M-RoPE positions follow it, each present only when the graph
        takes them.
        """
        device_refs = list(self.devices)
        inputs: OrderedDict[str, object] = OrderedDict(
            batch=ragged_batch_inputs(device_refs[0]),
            signals=signal_inputs(device_refs),
            kv=kv_params.get_symbolic_inputs(),
        )
        if self.vision_encoder is not None:
            # The hidden state is replicated across devices, so the merge runs
            # per replica against a per-device copy of the same embeddings.
            inputs["vision"] = vision_merge_inputs(
                device_refs,
                self.config.compute_dtype,
                self.config.hidden_size,
                rows_dim="vision_merged_seq_len",
            )
        if self.mrope_enabled:
            # One shared table drives every full-attention layer, so the
            # positions arrive once rather than per device.
            inputs["mrope"] = MRoPEPositions(
                position_ids=TensorType(
                    DType.int64,
                    shape=[3, "total_seq_len"],
                    device=device_refs[0],
                )
            )
        return inputs

    def input_types(
        self, kv_params: KVCacheParamInterface
    ) -> tuple[TensorType | BufferType, ...]:
        """Get input types for graph construction."""
        return tuple(tree.leaves(self.input_tree(kv_params)))

    def trace(
        self,
        *input_types: Type[Any],
        name: str | None = None,
        module: GraphModule | None = None,
        custom_extensions: Iterable[Path] = (),
        is_device_graph: bool = False,
    ) -> Graph:
        """Traces the language graph, whose inputs are the leaves of
        :meth:`input_tree` for :attr:`kv_params`.

        Adapts the flat graph inputs to :meth:`__call__`: splits the cache
        into its attention collections and recurrent state, and passes the
        vision and M-RoPE inputs when the graph takes them.

        Args:
            *input_types: The graph's input types.
            name: The name of the graph. Defaults to the class's qualified
                name.
            module: The MLIR module to add the graph to. Defaults to a new
                module.
            custom_extensions: Paths to custom kernel libraries the graph
                uses.
            is_device_graph: Whether to build a device graph.

        Returns:
            The traced :class:`~max.graph.Graph`.
        """
        treedef = tree.flatten(self.input_tree(self.kv_params))[1]

        def forward(*inputs: Value[Any]) -> DecoderOutputs:
            named = tree.unflatten(treedef, iter(inputs))
            assert isinstance(named, OrderedDict)
            batch, kv = named["batch"], named["kv"]
            state = tree.leaves(
                kv[STATE_CACHE_KEY], leaf=RecurrentStateInputsPerDevice
            )
            assert state, (
                "Qwen3.5 graph requires linear attention layers; the cache"
                " declared no recurrent state child"
            )
            vision = named.get("vision")
            mrope = named.get("mrope")
            return self(
                batch["tokens"].tensor,
                tree.leaves(kv[ATTN_CACHE_KEY], leaf=KVCacheInputsPerDevice),
                batch["return_n_logits"].tensor,
                batch["row_offsets"].tensor,
                [v.buffer for v in named["signals"]["signal"]],
                list(state),
                None
                if vision is None
                else [v.tensor for v in vision["image_embeddings"]],
                None
                if vision is None
                else [v.tensor for v in vision["image_token_indices"]],
                None if mrope is None else mrope["position_ids"].tensor,
            )

        return Graph(
            type(self).__qualname__ if name is None else name,
            forward,
            input_types=input_types,
            custom_extensions=list(custom_extensions),
            module=module,
            is_device_graph=is_device_graph,
        )
