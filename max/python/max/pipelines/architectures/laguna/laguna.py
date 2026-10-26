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

"""Implements the Laguna model.

Laguna is a Mixture-of-Experts decoder-only transformer with:

- GQA attention with per-head QK norm and rotary embeddings
- A softplus gate applied to the attention output, per head or per element
  depending on the config's ``gating``
- Sigmoid MoE routing with per-expert score-correction bias
- Gated MLP (SwiGLU) experts

Attention type varies per layer, from ``layer_types``: ``sliding_attention``
layers get their own KV cache leaf, sliding-window mask and RoPE table, while
``full_attention`` layers stay full causal on theirs, and a checkpoint with no
sliding layer keeps a single flat cache and table. The MLP type (dense prefix
vs sparse MoE) varies per layer too, from ``mlp_layer_types``, as does its
precision: a checkpoint can pack some layers' MLPs as NVFP4 and leave the rest
bfloat16.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Iterable, Mapping, Sequence
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
from max.nn.comm import Signals
from max.nn.comm.ep import EPBatchManager
from max.nn.data_parallelism import split_batch_replicated
from max.nn.embedding import VocabParallelEmbedding
from max.nn.kv_cache import (
    KVCacheInputsPerDevice,
    KVCacheParamInterface,
    KVCacheParams,
    MultiKVCacheParams,
    PagedCacheValues,
)
from max.nn.layer import LayerList, Module
from max.nn.linear import MLP as DenseMLP
from max.nn.linear import ColumnParallelLinear, Linear
from max.nn.moe import MoE, MoEQuantized, forward_moe_sharded_layers
from max.nn.norm import RMSNorm
from max.nn.rotary_embedding import RotaryEmbedding, YarnScalingParams
from max.nn.transformer import (
    DecoderOutputs,
    ReturnHiddenStates,
    forward_sequential_layers,
    logits_outputs,
)
from max.nn.transformer.distributed_transformer import (
    DistributedLogitsPostprocessMixin,
    forward_sharded_layers,
)
from max.pipelines.architectures.gemma4.layers.rotary_embedding import (
    ProportionalScalingParams,
)
from max.tree import Tree

from .layers.attention import LagunaAttention
from .layers.moe_gate import LagunaTopKRouter
from .layers.rotary_embedding import (
    LagunaRotaryEmbedding,
    LagunaYarnPartialRotaryEmbedding,
)
from .model_config import LagunaConfig


def _rotary_dim(rope_params: Mapping[str, Any], head_dim: int) -> int:
    """Returns how many of a head's lanes a RoPE table rotates."""
    return int(head_dim * float(rope_params.get("partial_rotary_factor", 1.0)))


class LagunaTransformerBlock(Module):
    """Laguna decoder block.

    Four things vary per layer: the attention type, which picks the mask, the
    KV cache leaf, and the RoPE table and rotary dim; the query-head count,
    which sizes ``q_proj``, ``o_proj`` and ``g_proj`` and so the GQA ratio (KV
    heads stay uniform); the MLP type, dense SwiGLU or EP-sharded sparse MoE;
    and whether that MLP is packed NVFP4. All four are fixed at construction,
    so a subgraph group never mixes them.
    """

    def __init__(
        self,
        config: LagunaConfig,
        layer_idx: int,
        rope: RotaryEmbedding,
        create_norm: Callable[..., RMSNorm],
        linear_cls: Callable[..., Linear],
        ep_manager: EPBatchManager | None = None,
        *,
        attention_type: str,
        num_attention_heads: int,
        kv_params: KVCacheParams,
        local_window_size: int,
        gating_mode: str,
        rotary_dim: int,
    ) -> None:
        super().__init__()
        self.devices = config.devices
        num_devices = len(config.devices)
        self.layer_idx = layer_idx
        self.attention_type = attention_type
        is_sliding = attention_type == "sliding_attention"

        mlp_type = (
            config.mlp_layer_types[layer_idx]
            if config.mlp_layer_types is not None
            else "sparse"
        )
        self.is_sparse_mlp = mlp_type == "sparse"

        # Laguna NVFP4 checkpoints keep attention in bf16 (it is in the quant
        # ignore list); only MLP/MoE are FP4. Quantize attention only when its
        # weights are actually a packed/low-precision dtype (auto-detected from
        # q_proj), NOT merely because attn_dtype matches the bf16 compute dtype.
        attn_dtype = config.attn_dtype or config.dtype
        attn_is_quantized = (
            config.quant_config is not None
            and attn_dtype
            not in (
                DType.bfloat16,
                DType.float16,
                DType.float32,
            )
        )
        attn_quant_config = config.quant_config if attn_is_quantized else None
        attn_linear_cls = linear_cls if attn_is_quantized else Linear

        self.self_attn = LagunaAttention(
            num_attention_heads=num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            hidden_size=config.hidden_size,
            kv_params=kv_params,
            layer_idx=layer_idx,
            is_sliding=is_sliding,
            local_window_size=local_window_size,
            gating_mode=gating_mode,
            rotary_dim=rotary_dim,
            dtype=attn_dtype,
            rope=rope,
            linear_cls=attn_linear_cls,
            devices=config.devices,
            scale=config.attention_multiplier,
            norm_dtype=config.norm_dtype or config.dtype,
            quant_config=attn_quant_config,
        )
        self.self_attn.sharding_strategy = ShardingStrategy.replicate(
            num_devices
        )
        self.self_attn_shards = self.self_attn.shard(config.devices)

        self.ep_manager = ep_manager
        if self.is_sparse_mlp:
            # Sparse and dense layers assign different (both Shardable) types
            # to the same attribute, so annotate the union.
            self.mlp: MoE | DenseMLP = self._get_moe(config, layer_idx)
            if ep_manager is not None:
                # Multi-GPU: expert-parallel sharding via the EP batch
                # manager. Each device owns a slice of the 256 experts;
                # the EP all-to-all dispatch routes tokens to experts.
                self.mlp.sharding_strategy = ShardingStrategy.expert_parallel(
                    num_devices
                )
            else:
                # Single-GPU: no EP. MoE class doesn't accept
                # ``replicate`` — only ``tensor_parallel`` or
                # ``expert_parallel``. With num_devices=1 TP collapses
                # to no-op (each "tensor parallel rank" gets all
                # experts), giving us the local-single-device path.
                self.mlp.sharding_strategy = ShardingStrategy.tensor_parallel(
                    num_devices
                )
            # MoE and dense shards both satisfy this callable-layer type; the
            # annotation keeps the dense-branch assignment below compatible.
            self.mlp_shards: Sequence[Callable[[TensorValue], TensorValue]] = (
                self.mlp.shard(config.devices)
            )
        else:
            self.mlp = self._get_dense_mlp(config, layer_idx)
            self.mlp.sharding_strategy = ShardingStrategy.replicate(num_devices)
            self.mlp_shards = self.mlp.shard(config.devices)

        self.input_layernorm = create_norm()
        self.input_layernorm.sharding_strategy = ShardingStrategy.replicate(
            num_devices
        )
        self.input_layernorm_shards = self.input_layernorm.shard(config.devices)

        self.post_attention_layernorm = create_norm()
        self.post_attention_layernorm.sharding_strategy = (
            ShardingStrategy.replicate(num_devices)
        )
        self.post_attention_layernorm_shards = (
            self.post_attention_layernorm.shard(config.devices)
        )

    def _get_moe(self, config: LagunaConfig, layer_idx: int) -> MoE:
        """Creates the sparse MoE block (sigmoid routing + shared experts + scaling).

        ``moe_dim`` here is ``moe_intermediate_size`` (the routed-expert
        intermediate dim from the HF config — NOT the dense-layer
        intermediate, ``intermediate_size_dense``).

        The routed experts may be packed NVFP4 or plain bfloat16;
        ``quant_config.mlp_quantized_layers``, read off the checkpoint, says
        which. An unquantized layer must also drop ``config.dtype``, which on a
        quantized checkpoint is the packed-FP4 storage dtype.
        """
        quant_config = config.quant_config
        is_quantized = (
            quant_config is not None
            and layer_idx in quant_config.mlp_quantized_layers
        )
        moe_cls = MoEQuantized if is_quantized else MoE
        return moe_cls(
            devices=config.devices,
            hidden_dim=config.hidden_size,
            num_experts=config.num_local_experts,
            num_experts_per_token=config.num_experts_per_tok,
            moe_dim=config.moe_intermediate_size,
            gate_cls=functools.partial(
                LagunaTopKRouter,
                norm_topk_prob=config.norm_topk_prob,
                gate_dtype=config.gate_dtype or config.dtype,
                correction_bias_dtype=(
                    config.correction_bias_dtype or DType.float32
                ),
                routed_scaling_factor=config.moe_routed_scaling_factor,
                router_logit_softcapping=config.moe_router_logit_softcapping,
            ),
            dtype=config.dtype if is_quantized else config.unquantized_dtype,
            quant_config=quant_config if is_quantized else None,
            # ``max.nn.moe`` keeps the shared experts unquantized only when
            # their dtype differs from the routed one; equal dtypes pack them
            # alongside, which is what a uniformly packed checkpoint wants.
            shared_experts_dtype=(
                quant_config.shared_experts_weight_dtype or config.dtype
                if is_quantized and quant_config is not None
                else None
            ),
            # Laguna has an always-on shared expert added to the routed
            # output. MAX MoE handles this when ``has_shared_experts=True``
            # — it materialises a ``self.shared_experts`` MLP and adds its
            # output to the routed sum.
            has_shared_experts=True,
            shared_experts_dim=config.shared_expert_intermediate_size,
            ep_batch_manager=self.ep_manager,
            ep_size=(
                self.ep_manager.config.n_gpus_per_node
                * self.ep_manager.config.n_nodes
                if self.ep_manager is not None
                else 1
            ),
        )

    def _get_dense_mlp(self, config: LagunaConfig, layer_idx: int) -> DenseMLP:
        """Builds the dense SwiGLU MLP a ``mlp_layer_types[i] == "dense"`` layer uses.

        Distinct from the routed-expert MLPs (which use
        ``moe_intermediate_size``). The dense MLP uses
        ``intermediate_size_dense`` from the HF config, and is packed NVFP4 only
        when the checkpoint shipped scales for it.
        """
        is_quantized = (
            config.quant_config is not None
            and layer_idx in config.quant_config.mlp_quantized_layers
        )
        return DenseMLP(
            dtype=config.dtype if is_quantized else config.unquantized_dtype,
            quantization_encoding=None,
            quant_config=config.quant_config if is_quantized else None,
            hidden_dim=config.hidden_size,
            feed_forward_length=config.intermediate_size_dense,
            devices=config.devices,
            linear_cls=Linear,
            has_bias=False,
            activation_function="silu",
        )

    def __call__(
        self,
        layer_idx: TensorValue,
        xs: list[TensorValue],
        signal_buffers: list[BufferValue],
        kv_collections: list[PagedCacheValues],
        freqs_cis: list[TensorValue],
        input_row_offsets: list[TensorValue],
        ep_inputs: list[TensorValue | BufferValue] | None = None,
    ) -> list[TensorValue]:
        """Runs the forward pass through the decoder block."""
        norm_xs = forward_sharded_layers(self.input_layernorm_shards, xs)

        # Replicated — no allreduce needed.
        attn_outs = [
            shard(
                layer_idx,
                norm_xs[i],
                kv_collections[i],
                freqs_cis[i],
                input_row_offsets[i],
            )
            for i, shard in enumerate(self.self_attn_shards)
        ]

        hs = [x + attn_out for x, attn_out in zip(xs, attn_outs, strict=True)]

        norm_outs = forward_sharded_layers(
            self.post_attention_layernorm_shards, hs
        )

        if self.is_sparse_mlp:
            if self.ep_manager is not None:
                # Multi-GPU EP path: dispatch via the EP batch manager.
                if ep_inputs is not None:
                    self.ep_manager.fetch_buffers(ep_inputs)
                mlp_outs = forward_moe_sharded_layers(
                    self.mlp_shards, norm_outs
                )
            else:
                # Single-GPU: each shard is a complete local MoE; run
                # it directly like a replicate-sharded module.
                mlp_outs = forward_sharded_layers(self.mlp_shards, norm_outs)
        else:
            # Dense MLP path (layer 0): replicate shards run per-device, no
            # allreduce.
            mlp_outs = forward_sharded_layers(self.mlp_shards, norm_outs)

        hs = [h + mlp_out for h, mlp_out in zip(hs, mlp_outs, strict=True)]

        return hs


class Laguna(DistributedLogitsPostprocessMixin, Module):
    """Laguna model supporting single-GPU and multi-GPU TP inference.

    Owns the per-layer schedule the blocks are built from: which attention type,
    cache leaf and RoPE table each layer takes, how many query heads it has, and
    which layers can share a decoder subgraph.
    """

    def __init__(self, config: LagunaConfig) -> None:
        super().__init__()
        self.config = config
        self.devices = config.devices
        self.num_devices = len(config.devices)
        self.ep_manager: EPBatchManager | None = None
        if config.ep_config is not None:
            self.ep_manager = EPBatchManager(config.ep_config)

        if config.model_quantization_encoding == QuantizationEncoding.GPTQ:
            raise NotImplementedError("GPTQ Laguna is not implemented yet")
        if (
            config.model_quantization_encoding is not None
            and config.model_quantization_encoding != QuantizationEncoding.GPTQ
        ):
            raise NotImplementedError("GGUFQ Laguna is not implemented yet")

        # Both cache leaves share head_dim, so either resolves the RoPE table.
        kv_params_by_type = self._kv_params_by_type(config)
        head_dim = kv_params_by_type["full_attention"].head_dim

        schedule = config.layer_schedule
        if schedule is None:
            raise ValueError(
                "LagunaConfig.layer_schedule is unset; it is populated by "
                "LagunaConfig.initialize_from_config."
            )

        # One RoPE table per attention type: the full-attention table is
        # YaRN-scaled and rotates only part of each head, the sliding one is
        # plain and full-rotary.
        rope_params_by_type: dict[str, Mapping[str, Any]] = {
            "full_attention": schedule.full_rope
        }
        if schedule.sliding_rope is not None:
            rope_params_by_type["sliding_attention"] = schedule.sliding_rope
        self._rope_by_type = {
            layer_type: self._build_rope(config, params, head_dim)
            for layer_type, params in rope_params_by_type.items()
        }
        self._rotary_dim_by_type = {
            layer_type: _rotary_dim(params, head_dim)
            for layer_type, params in rope_params_by_type.items()
        }
        self.rope_full = self._rope_by_type["full_attention"]
        self.rope_sliding = self._rope_by_type.get("sliding_attention")

        create_norm = functools.partial(
            RMSNorm,
            config.hidden_size,
            dtype=config.norm_dtype or DType.float32,
            eps=config.rms_norm_eps or 1e-6,
            multiply_before_cast=False,
        )

        linear_cls = functools.partial(Linear, quant_config=config.quant_config)

        num_layers = config.num_hidden_layers
        self._layer_types = list(schedule.layer_types)
        self._layer_heads = list(schedule.num_heads_per_layer)

        # Each layer addresses its cache leaf by an index counted within its
        # own attention type, not by its position in the model.
        type_counts: dict[str, int] = {}
        self._layer_idx_in_cache: list[int] = []
        for layer_type in self._layer_types:
            self._layer_idx_in_cache.append(type_counts.get(layer_type, 0))
            type_counts[layer_type] = type_counts.get(layer_type, 0) + 1

        self.layers = LayerList(
            [
                LagunaTransformerBlock(
                    config=config,
                    layer_idx=i,
                    rope=self._rope_by_type[self._layer_types[i]],
                    create_norm=create_norm,
                    linear_cls=linear_cls,
                    ep_manager=self.ep_manager,
                    attention_type=self._layer_types[i],
                    num_attention_heads=self._layer_heads[i],
                    kv_params=kv_params_by_type[self._layer_types[i]],
                    local_window_size=schedule.sliding_window or -1,
                    gating_mode=schedule.gating_mode,
                    rotary_dim=self._rotary_dim_by_type[self._layer_types[i]],
                )
                for i in range(num_layers)
            ]
        )

        self._subgraph_group_keys: list[tuple[str, int, str, bool]] = []
        self.subgraph_layer_groups: list[list[int]] = []
        if config.use_subgraphs:
            # A subgraph's signature is fixed by its first layer, so layers may
            # only share one when they agree on everything that shapes a block.
            mlp_types = config.mlp_layer_types or (["sparse"] * num_layers)
            quantized_mlps = (
                config.quant_config.mlp_quantized_layers
                if config.quant_config is not None
                else set()
            )
            groups: dict[tuple[str, int, str, bool], list[int]] = {}
            for i in range(num_layers):
                groups.setdefault(
                    (
                        self._layer_types[i],
                        self._layer_heads[i],
                        mlp_types[i],
                        i in quantized_mlps,
                    ),
                    [],
                ).append(i)
            self._subgraph_group_keys = list(groups)
            self.subgraph_layer_groups = list(groups.values())

        self.norm = create_norm()
        self.norm.sharding_strategy = ShardingStrategy.replicate(
            self.num_devices
        )
        self.norm_shards = self.norm.shard(config.devices)

        embedding_dtype = config.dtype
        if config.quant_config and config.quant_config.embedding_output_dtype:
            embedding_dtype = config.quant_config.embedding_output_dtype

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            embedding_dtype,
            config.devices,
        )
        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            embedding_dtype,
            devices=config.devices,
            tied_weight=(
                self.embed_tokens.weight if config.tie_word_embeddings else None
            ),
        )

        self.kv_params = config.kv_params
        self.return_logits = config.return_logits

    @staticmethod
    def _build_rope(
        config: LagunaConfig,
        rope_params: Mapping[str, Any],
        head_dim: int,
    ) -> RotaryEmbedding:
        """Builds one attention type's RoPE table from its HF entry.

        Args:
            config: The Laguna config.
            rope_params: One entry of the HF config's ``rope_parameters``.
            head_dim: Lanes per head, shared by both cache leaves.

        Returns:
            The rotary embedding for the layers of that attention type.

        Raises:
            KeyError: If a YaRN table omits ``factor`` or
                ``original_max_position_embeddings``.
        """
        theta = float(rope_params.get("rope_theta", 10000.0))
        partial_rotary_factor = float(
            rope_params.get("partial_rotary_factor", 1.0)
        )
        if rope_params.get("rope_type", "default") == "yarn":
            return LagunaYarnPartialRotaryEmbedding(
                dim=config.hidden_size,
                n_heads=config.num_attention_heads,
                theta=theta,
                max_seq_len=config.max_seq_len,
                head_dim=head_dim,
                interleaved=False,
                scaling_params=YarnScalingParams(
                    factor=float(rope_params["factor"]),
                    beta_fast=float(rope_params.get("beta_fast", 32.0)),
                    beta_slow=float(rope_params.get("beta_slow", 1.0)),
                    original_max_position_embeddings=int(
                        rope_params["original_max_position_embeddings"]
                    ),
                    truncate=True,
                ),
                partial_rotary_factor=partial_rotary_factor,
            )
        return LagunaRotaryEmbedding(
            dim=config.hidden_size,
            n_heads=config.num_attention_heads,
            theta=theta,
            max_seq_len=config.max_seq_len,
            head_dim=head_dim,
            interleaved=False,
            scaling_params=ProportionalScalingParams(
                partial_rotary_factor=partial_rotary_factor,
            ),
        )

    @staticmethod
    def _kv_params_by_type(config: LagunaConfig) -> dict[str, KVCacheParams]:
        """Maps each attention type to the cache leaf its layers address.

        Args:
            config: The Laguna config.

        Returns:
            A ``KVCacheParams`` per attention type present in the model.
        """
        if isinstance(config.kv_params, MultiKVCacheParams):
            leaves: dict[str, KVCacheParams] = {}
            for key, leaf in config.kv_params.children.items():
                assert isinstance(leaf, KVCacheParams)
                leaves[key] = leaf
            return leaves
        assert isinstance(config.kv_params, KVCacheParams)
        return {"full_attention": config.kv_params}

    def __call__(
        self,
        tokens: TensorValueLike,
        sliding_kv_collections: list[PagedCacheValues],
        full_kv_collections: list[PagedCacheValues],
        return_n_logits: TensorValue,
        input_row_offsets: TensorValue,
        signal_buffers: list[BufferValue],
        ep_inputs: list[TensorValue | BufferValue] | None = None,
        data_parallel_splits: TensorValue | None = None,
        host_input_row_offsets: TensorValue | None = None,
    ) -> DecoderOutputs:
        """Runs the forward pass through the model.

        Args:
            tokens: Input token IDs.
            sliding_kv_collections: Sliding-attention KV cache per device;
                empty when the model has no sliding layer.
            full_kv_collections: Full-attention KV cache per device.
            return_n_logits: Number of logits to return.
            input_row_offsets: Row offsets for ragged batching.
            signal_buffers: Signal buffers for allreduce.
            ep_inputs: Expert parallelism communication buffers.
            data_parallel_splits: Per-rank token counts for DP splitting.
            host_input_row_offsets: Host-side row offsets for DP splitting.
        """
        # Allreduce produces identical embeddings on all GPUs.
        h = self.embed_tokens(tokens, signal_buffers)

        # One table per attention type, each replicated per device.
        freqs_cis_by_type = {
            layer_type: [rope.freqs_cis.to(device) for device in self.devices]
            for layer_type, rope in self._rope_by_type.items()
        }

        input_row_offsets_list = ops.distributed_broadcast(
            input_row_offsets.to(self.devices[0]), signal_buffers
        )

        # Split replicated batch across GPUs for DP. With DP=1 (single-
        # replica, the default), there is nothing to split — every
        # device sees the same single batch, and we leave ``h`` /
        # ``input_row_offsets_list`` as the broadcast form. The donor
        # required DP > 1 by construction;
        # ``LagunaConfig.construct_kv_params`` now decouples DP degree
        # from device count, so this guard fires for both single-GPU
        # and multi-GPU runs.
        if (
            data_parallel_splits is not None
            and host_input_row_offsets is not None
            and self.config.data_parallel_degree > 1
        ):
            h, input_row_offsets_list = split_batch_replicated(
                self.devices,
                h,
                input_row_offsets_list,
                host_input_row_offsets.cast(DType.int64),
                data_parallel_splits,
            )

        def inputs_for_layer(idx: int, h: list[TensorValue]) -> list[Tree[Any]]:
            layer_type = self._layer_types[idx]
            kv_collections = (
                sliding_kv_collections
                if layer_type == "sliding_attention"
                else full_kv_collections
            )
            values: list[Tree[Any]] = [
                ops.constant(
                    self._layer_idx_in_cache[idx],
                    DType.uint32,
                    device=DeviceRef.CPU(),
                ),
                h,
                signal_buffers,
                kv_collections,
                freqs_cis_by_type[layer_type],
                input_row_offsets_list,
            ]
            if ep_inputs is not None:
                values.append(ep_inputs)
            return values

        def name_for_subgraph(group_idx: int) -> str:
            attention_type, heads, mlp_type, is_quantized = (
                self._subgraph_group_keys[group_idx]
            )
            flavor = (
                "sliding" if attention_type == "sliding_attention" else "full"
            )
            # Every part of the group key has to appear in the name too, or two
            # groups could collide on one subgraph.
            precision = "nvfp4_" if is_quantized else ""
            return f"laguna_{flavor}_{heads}h_{precision}{mlp_type}_block"

        h = forward_sequential_layers(
            list(self.layers),
            inputs_for_layer=inputs_for_layer,
            weight_prefix_for_layer=lambda i: f"layers.{i}.",
            subgraph_layer_groups=self.subgraph_layer_groups or None,
            name_for_subgraph=name_for_subgraph,
            initial_hidden_states=h,
        )

        # DP logit postprocessing: allgather last tokens, then lm_head.
        last_token_per_dev = [
            ops.gather(h[i], input_row_offsets_list[i][1:] - 1, axis=0)
            for i in range(len(self.devices))
        ]
        last_token_distributed = ops.allgather(
            last_token_per_dev, signal_buffers
        )
        norm_last_token = forward_sharded_layers(
            self.norm_shards, last_token_distributed
        )
        last_logits = ops.cast(
            self.lm_head(norm_last_token, signal_buffers)[0],
            DType.float32,
        )
        return logits_outputs(
            last_logits, None, None, ReturnHiddenStates.NONE, []
        )

    def input_types(
        self, kv_params: KVCacheParamInterface
    ) -> tuple[TensorType | BufferType, ...]:
        """Returns the input types for graph construction."""
        device_ref = self.devices[0]

        tokens_type = TensorType(
            DType.int64, shape=["total_seq_len"], device=device_ref
        )
        input_row_offsets_type = TensorType(
            DType.uint32, shape=["input_row_offsets_len"], device=device_ref
        )
        return_n_logits_type = TensorType(
            DType.int64, shape=["return_n_logits"], device=DeviceRef.CPU()
        )
        data_parallel_splits_type = TensorType(
            DType.int64,
            shape=["data_parallel_splits_len"],
            device=DeviceRef.CPU(),
        )
        host_input_row_offsets_type = TensorType(
            DType.uint32,
            shape=["input_row_offsets_len"],
            device=DeviceRef.CPU(),
        )

        kv_inputs = kv_params.get_symbolic_inputs()

        base_inputs: list[TensorType | BufferType] = [
            tokens_type,
            input_row_offsets_type,
            return_n_logits_type,
        ]
        # DP-N path (multi-replica): graph also takes DP splits + host row
        # offsets. With DP=1 (the only validated configuration; see arch.py)
        # the model's forward doesn't call ``split_batch_replicated`` and the
        # batch processor omits these, so the graph input list must match.
        if self.config.data_parallel_degree > 1:
            base_inputs += [
                data_parallel_splits_type,
                host_input_row_offsets_type,
            ]

        signals = Signals(devices=self.devices)
        signal_buffer_types = signals.input_types()
        flattened_kv_types = tree.leaves(kv_inputs)

        ep_input_types: list[TensorType | BufferType] = []
        if self.ep_manager is not None:
            ep_input_types = list(self.ep_manager.input_types())

        return tuple(
            base_inputs
            + signal_buffer_types
            + flattened_kv_types
            + ep_input_types
        )

    def trace(
        self,
        *input_types: Type[Any],
        name: str | None = None,
        module: GraphModule | None = None,
        custom_extensions: Iterable[Path] = (),
        is_device_graph: bool = False,
    ) -> Graph:
        """Traces the graph, whose inputs are :meth:`input_types`.

        Adapts the flat graph inputs to :meth:`__call__`: splits them into
        the batch, the data-parallel splits, a signal buffer per device, the
        KV cache and the expert-parallel buffers.

        Args:
            *input_types: The graph's input types, as :meth:`input_types`
                returns them.
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
        n_devices = len(self.devices)

        def forward(*inputs: Value[Any]) -> DecoderOutputs:
            inputs_iter = iter(inputs)
            tokens = next(inputs_iter)
            input_row_offsets = next(inputs_iter)
            return_n_logits = next(inputs_iter)
            data_parallel_splits = None
            host_input_row_offsets = None
            if self.config.data_parallel_degree > 1:
                data_parallel_splits = next(inputs_iter).tensor
                host_input_row_offsets = next(inputs_iter).tensor

            signal_buffers = [
                next(inputs_iter).buffer for _ in range(n_devices)
            ]
            num_kv_inputs = len(self.kv_params.flattened_kv_inputs())
            kv_inputs = [next(inputs_iter) for _ in range(num_kv_inputs)]
            if isinstance(self.kv_params, MultiKVCacheParams):
                # The tree's declaration order is the unflattening order, and
                # swapping the two leaves would run every layer against the
                # wrong cache without failing anything.
                children = list(self.kv_params.children)
                assert children == ["sliding_attention", "full_attention"], (
                    f"Unexpected Laguna KV tree order: {children}"
                )
                sliding_kv_collections, full_kv_collections = (
                    self.kv_params.unflatten_basic_kv_tree(iter(kv_inputs))
                )
            else:
                sliding_kv_collections = []
                full_kv_collections = tree.leaves(
                    self.kv_params.unflatten_kv_inputs(iter(kv_inputs)),
                    leaf=KVCacheInputsPerDevice,
                )
            ep_inputs: list[TensorValue | BufferValue] = []
            for value in inputs_iter:
                assert isinstance(value, TensorValue | BufferValue)
                ep_inputs.append(value)

            return self(
                tokens.tensor,
                sliding_kv_collections,
                full_kv_collections,
                return_n_logits.tensor,
                input_row_offsets.tensor,
                signal_buffers,
                ep_inputs,
                data_parallel_splits,
                host_input_row_offsets,
            )

        return Graph(
            type(self).__qualname__ if name is None else name,
            forward,
            input_types=input_types,
            custom_extensions=list(custom_extensions),
            module=module,
            is_device_graph=is_device_graph,
        )
