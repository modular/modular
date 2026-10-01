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
"""Graph assembly for GLM-5.3-Flash (``glm5_next``).

This module owns the parts of the decoder that belong to no single sublayer:
the hybrid attention schedule, the residual-stream plumbing around it, the MLP
choice per layer, and the head. The four sublayers are implemented by their own lanes and satisfy
``Glm5NextDecoderLayer``'s ``Glm5NextSublayer`` protocol structurally, so this
file constructs none of them: see ``tasks/glm53-scope/contracts/`` for the two
published interfaces.

The schedule, verified against the checkpoint's weight index rather than
inferred from the config::

    idx  0  1  2  3  4  5  6  7  8  9 10 11 ...  43 44 | 45
         K  K  K  D  K  K  K  D  K  K  K  D      D  K  | MTP (D, no mHC)
    mlp  d  d  d  S  S  S  S  S  S  S  S  S      S  S  | S

    K = KDA linear attention (34)   D = sparse MLA + DSA indexer (11 + MTP)
    d = dense MLP (3)               S = MoE (43, including MTP)

Two structural facts drive the assembly:

* **The residual carries four streams, not one.** ``[tokens, hc_mult, hidden]``
  is created by :func:`~.layers.hyper_connection.expand_streams` and collapsed
  at the end by :func:`~.layers.hyper_connection.mean_collapse_streams`, an
  *unweighted* mean -- GLM's one change from DeepSeek-V4, which learns that
  collapse. There is consequently no ``hc_head`` weight in the checkpoint, and
  cloning DeepSeek-V4's learned head would try to load three tensors that do
  not exist.
* **The MTP draft layer at index 45 is the one structural exception.** It is
  sparse MLA with its own indexer and its own MoE block, and it has no
  hyper-connections: its residual is a plain add.
"""

from __future__ import annotations

import functools
from collections.abc import Sequence
from typing import Any

from max.dtype import DType
from max.graph import (
    BufferType,
    BufferValue,
    DeviceRef,
    ShardingStrategy,
    TensorType,
    TensorValue,
    ops,
)
from max.nn.comm import Signals
from max.nn.comm.ep import EPBatchManager
from max.nn.embedding import VocabParallelEmbedding
from max.nn.kv_cache import (
    KVCacheParamInterface,
    PagedCacheValues,
    RecurrentStateInputsPerDevice,
)
from max.nn.layer import LayerList, Module
from max.nn.linear import ColumnParallelLinear
from max.nn.norm import RMSNorm
from max.nn.transformer.transformer import forward_sequential_layers
from max.pipelines.architectures.deepseekV3.deepseekV3 import (
    deepseek_logits_postprocess,
)
from max.pipelines.architectures.deepseekV3_2.layers import (
    DeepseekV3_2TopKRouter,
)
from max.tree import Tree

from .layers.decoder import Glm5NextDecoderLayer
from .layers.hyper_connection import expand_streams, mean_collapse_streams
from .layers.kda import kda_sublayer_inputs
from .layers.kimi_delta_attention import Glm5NextKdaSublayer
from .layers.mlp import Glm5NextMLP, Glm5NextMlpSublayer, Glm5NextMoE
from .layers.sparse_mla import (
    Glm5NextSparseMLASublayer,
    SparseMLASublayerInputs,
    nope_pad_rotary,
)
from .model_config import SPARSE_ATTENTION, Glm5NextConfig
from .state_cache import TAIL_RING_LEAF_ID, leaf_inputs

__all__ = ["Glm5Next"]


class Glm5Next(Module):
    """The GLM-5.3-Flash language model graph.

    Assembles the period-4 hybrid schedule and the four-stream residual. The
    attention sublayers and the hyper-connection sites are constructed through
    the seam types, so this class is complete once the four lanes land their
    modules and needs no edit to accommodate them.
    """

    def __init__(self, config: Glm5NextConfig) -> None:
        super().__init__()
        self.config = config

        if not config.layer_types:
            raise ValueError(
                "Glm5NextConfig.layer_types is empty; the hybrid schedule "
                "cannot be assembled. Glm5NextConfig.resolve_layer_types "
                "reads it from the checkpoint."
            )

        embedding_output_dtype = config.compute_dtype
        if embedding_output_dtype == DType.uint8:
            embedding_output_dtype = DType.bfloat16
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            dtype=embedding_output_dtype,
            devices=config.devices,
            quantization_encoding=None,
        )

        # One manager per model, shared by every MoE layer: it owns the
        # dispatch/combine buffers the EP kernels communicate through, so a
        # per-layer instance would allocate a set nobody else can see.
        self.ep_manager: EPBatchManager | None = None
        if config.ep_config is not None:
            self.ep_manager = EPBatchManager(config.ep_config)

        # Populated by the lane modules through `attach_layers`. Kept as an
        # explicit step rather than constructed here so that this file does not
        # have to import four lanes' modules -- five engineers are editing this
        # package concurrently, and an import here would serialise them.
        self.layers = LayerList([])

        self.norm = RMSNorm(
            config.hidden_size,
            dtype=config.compute_dtype,
            eps=config.rms_norm_eps,
        )
        # The final norm runs per device: `ColumnParallelLinear` wants the full
        # hidden vector on each rank, and the norm is replicated, not sharded.
        self.norm.sharding_strategy = ShardingStrategy.replicate(
            len(config.devices)
        )
        self.norm_shards = self.norm.shard(config.devices)
        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            dtype=config.compute_dtype,
            devices=config.devices,
            tied_weight=(
                self.embed_tokens.weight if config.tie_word_embeddings else None
            ),
        )

    # ------------------------------------------------------------- schedule

    def expand_streams(self, hidden: TensorValue) -> TensorValue:
        """Broadcasts ``[T, hidden]`` into the ``[T, hc_mult, hidden]`` residual."""
        return expand_streams(hidden, hc_mult=self.config.hc_mult)

    def collapse_streams(self, streams: TensorValue) -> TensorValue:
        """Collapses the streams back to ``[T, hidden]`` by unweighted mean."""
        return mean_collapse_streams(streams)

    def layer_type(self, layer_idx: int) -> str:
        """Returns the attention family of ``layer_idx``.

        The MTP draft layer sits at ``num_hidden_layers`` and beyond, past the
        end of ``layer_types``; it is sparse MLA.
        """
        if layer_idx >= self.config.num_hidden_layers:
            return SPARSE_ATTENTION
        return self.config.layer_types[layer_idx]

    def uses_hyper_connections(self, layer_idx: int) -> bool:
        """Whether ``layer_idx`` has mHC sites rather than a plain residual add.

        Layers 0-44 do; the MTP draft layer at 45 does not. ``hc_attn_fn``
        appears 45 times in the checkpoint index while ``input_layernorm``
        appears 46 times, which is how this is established.
        """
        return self.config.mhc and layer_idx < self.config.num_hidden_layers

    def uses_moe(self, layer_idx: int) -> bool:
        """Whether ``layer_idx`` has an MoE block rather than a dense MLP.

        Layers 0-2 are dense at width ``intermediate_size`` (12288); 3 onward,
        the MTP layer included, are MoE at ``moe_intermediate_size`` (2048)
        with one shared expert and eight of 288 routed.
        """
        return layer_idx >= self.config.first_k_dense_replace

    def subgraph_layer_groups(self) -> list[list[int]]:
        """Groups the decoder layers that can share one compiled subgraph.

        A layer's subgraph signature is fixed by two things: which attention
        family it belongs to, because that decides its per-step bundle, and
        whether its feed-forward block is dense or MoE. Both come off the
        schedule rather than off the period, so a sibling that retunes either
        regroups itself. Nothing else varies -- every layer takes and returns
        the same ``hc_mult``-wide residual, so unlike DeepSeek-V3.2 no layer
        has to be peeled to keep a group's arity uniform.

        DeepSeek-V3.2 additionally splits its MoE layers on ``indexer_types``,
        because a ``"shared"`` layer declares no indexer weights.
        GLM-5.3-Flash's ``indexer_types`` is ``"full"`` on all 45 layers, so
        that split would produce one empty group and is not carried here.

        Groups of one are dropped: a subgraph with a single call site costs a
        boundary and saves no elaboration.

        Returns:
            One list of layer indices per group, or empty when subgraphs are
            off. The MTP draft layer is never included -- it is the one layer
            without hyper-connections, so its residual is rank 2 where every
            other layer's is rank 3.
        """
        if not self.config.use_subgraphs:
            return []
        groups: dict[tuple[str, bool], list[int]] = {}
        for layer_idx in range(self.config.num_hidden_layers):
            key = (self.layer_type(layer_idx), self.uses_moe(layer_idx))
            groups.setdefault(key, []).append(layer_idx)
        return [group for group in groups.values() if len(group) > 1]

    def build_layers(self) -> LayerList:
        """Constructs the decoder stack from the schedule.

        One :class:`~.layers.decoder.Glm5NextDecoderLayer` per index, each
        holding the attention sublayer its ``layer_types`` entry names and the
        feed-forward sublayer its index implies. The MTP draft layer, when
        built, is sparse MLA with ``mhc=False``.

        The rotary table is built **once and shared by every sparse layer**.
        GLM-5.3-Flash has no rotary embedding, but the 576-wide latent geometry
        carries 64 rotary columns and the fused ops read that width off
        ``freqs_cis.shape[1]``, so a table of that shape has to exist. Its
        values are irrelevant -- every column it rotates is zero -- but its
        leading dimension is ``max_seq_len``, which is 134 MB at 1M context.
        Once, not eleven times.
        """
        config = self.config
        rope = (
            nope_pad_rotary(config)
            if SPARSE_ATTENTION in config.layer_types
            else None
        )

        # Heterogeneous by construction: each layer's bundle type is its own
        # lane's, which is what `Glm5NextDecoderLayer`'s generics are for.
        layers: list[Glm5NextDecoderLayer[Any, Any]] = []
        for layer_idx in range(config.num_hidden_layers):
            if self.layer_type(layer_idx) == SPARSE_ATTENTION:
                assert rope is not None
                self_attn: Module = Glm5NextSparseMLASublayer(
                    config,
                    layer_idx,
                    rope=rope,
                )
            else:
                self_attn = Glm5NextKdaSublayer(config, layer_idx)
            layers.append(
                Glm5NextDecoderLayer(
                    config,
                    layer_idx,
                    self_attn=self_attn,
                    mlp=self.build_mlp(layer_idx),
                )
            )
        return LayerList(layers)

    def build_mlp(self, layer_idx: int) -> Module:
        """Returns the feed-forward sublayer for ``layer_idx``.

        Dense at ``intermediate_size`` below ``first_k_dense_replace``, MoE at
        ``moe_intermediate_size`` at and above it. Both clamp before the
        activation; see :mod:`.layers.mlp` for why none of MAX's three existing
        clamped-SwiGLU paths can be reused.
        """
        config = self.config
        mlp = self._mlp_module(layer_idx)
        return Glm5NextMlpSublayer(mlp, config.devices, self.ep_manager)

    def _mlp_module(self, layer_idx: int) -> Glm5NextMLP | Glm5NextMoE:
        """Returns the unwrapped dense MLP or MoE block for ``layer_idx``.

        Mirrors DeepSeek-V3.2's construction, because GLM-5.3-Flash's MoE *is*
        DeepSeek's apart from the activation: the same ``noaux_tc`` sigmoid
        router with a learned correction bias, the same one shared expert, the
        same float32 router math. Only ``gated_activation_fn`` differs, and
        :class:`~.layers.mlp.Glm5NextMoE` supplies it.

        Every MLP in this checkpoint is FP8 -- the three dense ones included --
        so ``mlp_dtype`` follows the resolved encoding rather than the compute
        dtype. Declaring the dense MLPs at the compute dtype instead is the
        first thing weight loading rejects.
        """
        config = self.config
        quant_cfg = config.quant_config
        quantized = (
            quant_cfg is not None
            and layer_idx in quant_cfg.mlp_quantized_layers
        )
        mlp_dtype = config.dtype if quantized else config.compute_dtype
        layer_quant_config = quant_cfg if quantized else None

        if not self.uses_moe(layer_idx):
            dense_mlp = Glm5NextMLP(
                dtype=mlp_dtype,
                quantization_encoding=None,
                hidden_dim=config.hidden_size,
                feed_forward_length=config.intermediate_size,
                devices=config.devices,
                quant_config=layer_quant_config,
                swiglu_limit=config.swiglu_limit,
            )
            # Mirrors DeepSeek-V3.2: the dense MLP is replicated unless EP is
            # reducing through allreduce, in which case it is tensor-parallel.
            # A strategy has to be set even at one device -- the sublayer
            # wrapper produces one shard per device only via `shard()`.
            if config.ep_config is not None and config.ep_config.use_allreduce:
                dense_mlp.sharding_strategy = ShardingStrategy.tensor_parallel(
                    len(config.devices)
                )
            else:
                dense_mlp.sharding_strategy = ShardingStrategy.replicate(
                    len(config.devices)
                )
            return dense_mlp

        ep_size = (
            config.ep_config.n_gpus_per_node * config.ep_config.n_nodes
            if config.ep_config is not None
            else 1
        )
        moe = Glm5NextMoE(
            devices=config.devices,
            hidden_dim=config.hidden_size,
            num_experts=config.n_routed_experts,
            num_experts_per_token=config.num_experts_per_tok,
            moe_dim=config.moe_intermediate_size,
            gate_cls=functools.partial(
                DeepseekV3_2TopKRouter,
                routed_scaling_factor=config.routed_scaling_factor,
                scoring_func=config.scoring_func,
                topk_method=config.topk_method,
                n_group=config.n_group,
                topk_group=config.topk_group,
                norm_topk_prob=config.norm_topk_prob,
                # The router runs in float32 and its weight is not quantized:
                # `modules_to_not_convert` excludes the gate on every GLM-5.x
                # checkpoint, and `moe_router_dtype` is float32.
                gate_dtype=DType.bfloat16,
                correction_bias_dtype=config.correction_bias_dtype,
            ),
            # Bound: the MoE constructs the shared expert through `mlp_cls`
            # and does not know about `swiglu_limit`.
            mlp_cls=functools.partial(
                Glm5NextMLP, swiglu_limit=config.swiglu_limit
            ),
            has_shared_experts=True,
            shared_experts_dim=config.n_shared_experts
            * config.moe_intermediate_size,
            dtype=mlp_dtype,
            ep_size=ep_size,
            ep_batch_manager=self.ep_manager,
            apply_router_weight_first=False,
            quant_config=layer_quant_config,
            shared_experts_dtype=(
                quant_cfg.shared_experts_dtype(mlp_dtype)
                if quant_cfg is not None
                else config.compute_dtype
            ),
            swiglu_limit=config.swiglu_limit,
        )
        # Mirrors DeepSeek-V3: expert-parallel when EP is on, tensor-parallel
        # otherwise. Left unset entirely, the sublayer wrapper falls back to a
        # single shard and the per-device forward sees one expert block for N
        # inputs.
        num_devices = len(config.devices)
        if num_devices > 1:
            if ep_size > 1:
                moe.sharding_strategy = ShardingStrategy.expert_parallel(
                    num_devices
                )
            else:
                moe.sharding_strategy = ShardingStrategy.tensor_parallel(
                    num_devices
                )
        return moe

    def attach_layers(self, layers: LayerList) -> None:
        """Installs the constructed decoder layers.

        Args:
            layers: One layer per index, ``num_hidden_layers`` plus
                ``num_nextn_predict_layers`` when the MTP draft is built.

        Raises:
            ValueError: If the count does not match the schedule.
        """
        expected = (
            self.config.num_hidden_layers,
            self.config.num_hidden_layers
            + self.config.num_nextn_predict_layers,
        )
        if len(layers) not in expected:
            raise ValueError(
                f"Expected {expected[0]} layers, or {expected[1]} with the MTP "
                f"draft layer, got {len(layers)}."
            )
        self.layers = layers

    # ------------------------------------------------------------ graph I/O

    def input_types(
        self, kv_params: KVCacheParamInterface
    ) -> list[TensorType | BufferType]:
        """Declares the graph's inputs, in the order ``execute`` supplies them.

        Five groups, and the order is the contract:

        1. ``tokens``, device row offsets, host row offsets,
           ``return_n_logits``, ``data_parallel_splits``
        2. the allreduce signal buffers
        3. both KV cache leaves, flattened -- ``"mla"`` then ``"indexer"``
        4. ``batch_context_lengths``, one per device

        The sparse layers' cache indices are graph *constants* rather than
        inputs, matching ``deepseekV3_2.py:840`` -- they are fixed by the
        schedule, so nothing has to supply them per step.

        Groups 1-4 are exactly ``DeepseekV3Inputs.buffers``. The recurrent
        state needs no group of its own: it is a child of the multi-cache, so
        its pools and row ids are already inside group 3.
        """
        device = self.config.devices[0]
        types: list[TensorType | BufferType] = [
            TensorType(DType.int64, shape=["total_seq_len"], device=device),
            TensorType(
                DType.uint32, shape=["input_row_offsets_len"], device=device
            ),
            TensorType(
                DType.uint32,
                shape=["input_row_offsets_len"],
                device=DeviceRef.CPU(),
            ),
            TensorType(
                DType.int64, shape=["return_n_logits"], device=DeviceRef.CPU()
            ),
        ]
        # `data_parallel_splits` and `batch_context_lengths` are declared even
        # though this graph does not read them yet: keeping the prefix
        # identical to `DeepseekV3Inputs.buffers` means the inherited input
        # preparation stays usable and only the GLM-specific groups are
        # appended. Dropping them instead put the two out of step by exactly
        # their count, which surfaces as an opaque arity error at execute.
        types.append(
            TensorType(
                DType.int64,
                shape=[self.config.data_parallel_degree + 1],
                device=DeviceRef.CPU(),
            )
        )
        types.extend(Signals(devices=self.config.devices).input_types())
        types.extend(kv_params.flattened_kv_inputs())
        types.extend(
            TensorType(DType.int32, shape=[1], device=DeviceRef.CPU())
            for _ in self.config.devices
        )
        if self.ep_manager is not None:
            types.extend(self.ep_manager.input_types())
        return types

    def __call__(
        self,
        tokens: TensorValue,
        signal_buffers: list[BufferValue],
        mla_kv_collections: list[PagedCacheValues],
        indexer_kv_collections: list[PagedCacheValues],
        return_n_logits: TensorValue,
        input_row_offsets: TensorValue,
        state: list[RecurrentStateInputsPerDevice[TensorValue, BufferValue]],
        ep_inputs: Sequence[Any] | None = None,
    ) -> tuple[TensorValue, ...]:
        """Runs the decoder and returns the logits tuple.

        The residual is widened to ``hc_mult`` streams once after the embedding
        and collapsed once before the final norm; every layer takes and returns
        the widened form. Both attention families receive rank-2 sequences --
        the widening is entirely inside the decoder layer's two mHC sites.
        """
        config = self.config
        devices = config.devices

        if self.ep_manager is not None:
            assert ep_inputs is not None, (
                "ep_config is set, so the EP communication buffers must be "
                "passed through from the graph inputs."
            )
        # Handed to every feed-forward sublayer rather than bound once on the
        # shared manager here: the MoE shards read the buffers off it while
        # they are traced, and under subgraphs that trace happens inside a
        # subgraph that cannot reference an outer-graph value.
        mlp_inputs = list(ep_inputs) if ep_inputs is not None else []

        h = self.embed_tokens(tokens, signal_buffers)
        row_offsets = ops.distributed_broadcast(
            input_row_offsets.to(devices[0]), signal_buffers
        )
        streams = [expand_streams(x, hc_mult=config.hc_mult) for x in h]

        kda_inputs = kda_sublayer_inputs(
            kda_layers=config.kda_layers,
            state=state,
            signal_buffers=signal_buffers,
            input_row_offsets=row_offsets,
        )

        # `layer_idx` for a sparse layer is its index within the *cached
        # subset*: only the sparse layers hold a KV cache, so model layer 3 is
        # cache layer 0.
        ring = [leaf_inputs(device, TAIL_RING_LEAF_ID) for device in state]
        sparse_inputs = {
            layer_idx: SparseMLASublayerInputs(
                signal_buffers=signal_buffers,
                input_row_offsets=row_offsets,
                mla_kv_collections=mla_kv_collections,
                indexer_kv_collections=indexer_kv_collections,
                # The cache index, not the model index: only the sparse layers
                # hold a KV cache, so model layer 3 is cache layer 0.
                layer_idx=ops.constant(
                    cache_idx, DType.uint32, device=DeviceRef.CPU()
                ),
                # Every sparse layer shares one ring pool per device and
                # differs only in the row it runs in. `cache_idx` is the
                # ring's own layer set -- the indexer cache's -- so it is the
                # column to read; the model index would bind one layer's ring
                # to another and pass every shape check.
                tail_pools=[leaf.pool for leaf in ring],
                tail_row_ids=[leaf.live_row_id(cache_idx) for leaf in ring],
            )
            for cache_idx, layer_idx in enumerate(
                config.sparse_attention_layers
            )
        }

        def inputs_for_layer(
            layer_idx: int, streams: list[TensorValue]
        ) -> list[Tree[Any]]:
            attn_inputs = (
                sparse_inputs[layer_idx]
                if self.layer_type(layer_idx) == SPARSE_ATTENTION
                else kda_inputs[layer_idx]
            )
            return [streams, attn_inputs, mlp_inputs]

        streams = forward_sequential_layers(
            list(self.layers),
            inputs_for_layer=inputs_for_layer,
            initial_hidden_states=streams,
            subgraph_layer_groups=self.subgraph_layer_groups(),
            name_for_subgraph=lambda group: f"glm5_next_block_{group}",
            weight_prefix_for_layer=lambda idx: f"layers.{idx}.",
        )

        hs = [mean_collapse_streams(s) for s in streams]
        return deepseek_logits_postprocess(
            h=hs,
            input_row_offsets=row_offsets,
            all_logits_input_row_offsets=None,
            return_n_logits=return_n_logits,
            norm_shards=self.norm_shards,
            lm_head=self.lm_head,
            signal_buffers=signal_buffers,
            devices=devices,
            is_data_parallel_attention=config.data_parallel_degree > 1,
            return_logits=config.return_logits,
            return_hidden_states=config.return_hidden_states,
        )
