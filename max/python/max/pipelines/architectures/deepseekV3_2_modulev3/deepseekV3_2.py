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
from collections.abc import Callable, Sequence
from typing import ClassVar

from max import functional as F
from max import tree
from max.driver import CPU
from max.dtype import DType
from max.graph import DeviceRef, TensorValue, ops
from max.nn import Module, as_subgraph
from max.nn.common_layers.embedding import VocabParallelEmbedding
from max.nn.common_layers.kv_cache import PagedCacheValues
from max.nn.common_layers.linear import ColumnParallelLinear
from max.nn.common_layers.mesh_axis import DP, TP
from max.nn.common_layers.quant_moe import QuantizedMoE
from max.nn.common_layers.rotary_embedding import RotaryEmbedding
from max.nn.kv_cache import KVCacheInputsPerDevice, KVCacheParamInterface
from max.nn.legacy.comm.ep import EPBatchManager, EPCommBuffers
from max.nn.outputs import ReturnHiddenStates, ReturnLogits
from max.nn.rope_scaling import DeepseekYarnRopeScalingParams
from max.nn.sequential import ModuleList
from max.pipelines.lib import ModuleV3Outputs
from max.pipelines.lib.role_contracts import DecoderInputs, DecoderOutputs
from max.pipelines.speculative.driver import (
    gather_variable_rows,
)
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
from ..deepseekV3_modulev3.deepseekV3 import (
    gather_last_tokens,
    split_replicated_batch,
)
from .layers.rms_norm import MultiplyBeforeCastRMSNorm
from .layers.transformer_block import DeepseekV3_2TransformerBlock
from .model_config import DeepseekV3_2Config, indexer_cache_layer_map


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


def replica_size(mesh: DeviceMesh) -> int:
    """Returns the devices one data-parallel replica spans.

    It is the ``tp`` axis of a ``(dp, tp)`` mesh, else one device.
    """
    return mesh.axis_size(TP) if TP in mesh.axis_names else 1


def _replica_rows_on_first_device(t: Tensor, mesh: DeviceMesh) -> Tensor:
    """Concatenates each replica's rows on the mesh's first device.

    ``t`` is replicated over the ``tp`` axis; each replica's first device's
    copy moves to the first device, in replica order.
    """
    group = replica_size(mesh)
    first = DeviceRef.from_device(mesh.devices[0])
    rows = [
        ops.transfer_to(TensorValue(t.local_shards[r * group]), first)
        for r in range(mesh.num_devices // group)
    ]
    return Tensor.from_graph_value(ops.concat(rows, axis=0))


def replica_context_lengths(
    batch_context_lengths: Sequence[Tensor], mesh: DeviceMesh
) -> Tensor:
    """Places the host context lengths on a host mirror of ``mesh``.

    The lengths come one per replica, or, on a ``(dp, tp)`` mesh, one per
    device, each holding its replica's (as the decoder's
    ``batch_context_lengths`` input types declare them). Of ``count``
    lengths, device ``i`` reads entry ``i // (mesh.num_devices // count)``.
    The host mesh has ``mesh``'s shape and axis names, so the attention
    layers address it like their device mesh.
    """
    count = len(batch_context_lengths)
    if mesh.num_devices % count:
        raise ValueError(
            f"{count} context lengths do not split the "
            f"{mesh.num_devices} devices of {mesh}"
        )
    group = mesh.num_devices // count
    cpu_mesh = DeviceMesh(
        tuple(CPU() for _ in range(mesh.num_devices)),
        mesh.mesh_shape,
        mesh.axis_names,
    )
    return Tensor.from_shard_values(
        [
            TensorValue(batch_context_lengths[i // group])
            for i in range(mesh.num_devices)
        ],
        DeviceMapping(cpu_mesh, (Replicated(),) * mesh.ndim),
    )


def rotary_embedding(config: DeepseekV3_2Config) -> RotaryEmbedding:
    """Returns the rotary embedding the config's layers attend with."""
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
        return DeepseekYarnRotaryEmbedding(
            dim=config.qk_rope_head_dim,
            n_heads=config.num_attention_heads,
            theta=config.rope_theta,
            max_seq_len=config.max_position_embeddings,
            device=config.devices[0].to_device(),
            interleaved=config.rope_interleave,
            scaling_params=scaling_params,
        )
    # GLM-5.x declares rope_type "default"; only DeepSeek proper uses
    # YaRN. Sized by ``max_seq_len`` rather than the model's
    # ``max_position_embeddings`` (1M for GLM): plain RoPE frequencies
    # depend only on position, so the shorter table is equivalent and
    # avoids materializing hundreds of MB of unused rows.
    return RotaryEmbedding(
        dim=config.qk_rope_head_dim,
        n_heads=config.num_attention_heads,
        theta=config.rope_theta,
        max_seq_len=config.max_seq_len,
        device=config.devices[0].to_device(),
        head_dim=config.qk_rope_head_dim,
        interleaved=config.rope_interleave,
    )


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
        tuple[Tensor, ...],
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

        self.rope = rotary_embedding(config)

        # One device per replica shards the vocabulary over the ``dp`` axis;
        # replicas of several devices shard it over their ``tp`` axis.
        tp_axis = (
            DP
            if config.data_parallel_degree > 1
            and (config.mesh is None or TP not in config.mesh.axis_names)
            else None
        )
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
        # A unified spec-decode graph sets these as the graph-API model does:
        # it reads VARIABLE logits and every token's normalized hidden state,
        # and skips the last-token projection it never uses.
        self.return_logits = ReturnLogits.LAST_TOKEN
        self.return_hidden_states = ReturnHiddenStates.NONE
        self.emit_last_token_logits = True

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
    ) -> tuple[Tensor, ...]:
        """Runs the embedding and decoder stack over the tokens."""
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
                Sharded(0) if name == DP else Replicated()
                for name in self.mesh.axis_names
            )
            h, input_row_offsets = split_replicated_batch(
                h,
                input_row_offsets,
                input_row_offsets_i64,
                data_parallel_splits,
                DeviceMapping(self.mesh, batch_placement),
                group_size=replica_size(self.mesh),
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
        indexer_slots = indexer_cache_layer_map(
            self.config.indexer_types, len(self.layers)
        )
        # Each block hands the next its input norm, fused into its post-MLP
        # all-gather, as the graph-API model does. Only the embeddings are
        # normalized here.
        carry = self.layers[0].carry_next_input_norm
        norm_h = self.layers[0].input_layernorm(h) if carry else None
        last = len(self.layers) - 1
        for idx, layer in enumerate(self.layers):
            layer_idx_tensor = F.constant(idx, DType.uint32, device=CPU())
            indexer_slot_tensor = F.constant(
                indexer_slots[idx], DType.uint32, device=CPU()
            )
            call: Callable[..., tuple[Tensor, ...]] = layer
            if isinstance(layer.mlp, QuantizedMoE):
                group = "shared" if layer.self_attn.skip_topk else "full"
                call = as_subgraph(layer, name=f"moe_block_{group}")
            elif self.config.decode_context_parallel_degree > 1:
                # The memory planner gives the top-level graph one pool that
                # lives for the whole forward, and an inline block's
                # transients land in it. Under DCP every rank attends all
                # heads, so a dense block's peak is ~14 GiB at a 32K-token
                # prefill: inline, that is held through all 78 layers on top
                # of each MoE block's own pool, and the step no longer fits.
                # As a subgraph call it is planned and freed per call.
                call = as_subgraph(layer, name="dense_block")
            # A block takes the next block's weight as an argument, since
            # a shared subgraph resolves only its own weights.
            next_input_gamma = (
                self.layers[idx + 1].input_layernorm.weight.cast(h.dtype)
                if carry and idx < last
                else None
            )
            h, topk_indices, *rest = call(
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
                indexer_slot_tensor,
                norm_x=norm_h,
                next_input_gamma=next_input_gamma,
            )
            norm_h = rest[0] if rest else None

        self._check_output_modes()
        outputs: list[Tensor] = []
        if self.emit_last_token_logits:
            last_token_h = gather_last_tokens(h, input_row_offsets)
            # Replicas of one device gather their rows before the vocabulary
            # projection; replicas of several gather their logits after it
            # (see :func:`project_to_logits`).
            if self.config.data_parallel_degree > 1 and (
                self.mesh is None or replica_size(self.mesh) == 1
            ):
                last_token_h = F.allgather(last_token_h, mesh_axis=DP)
            outputs.append(self._project(last_token_h))
        if self.return_logits is ReturnLogits.VARIABLE:
            outputs.extend(
                self._variable_logits(h, input_row_offsets, return_n_logits)
            )
        if self.return_hidden_states is ReturnHiddenStates.ALL_NORMALIZED:
            outputs.append(self.norm(h))
        return tuple(outputs)

    def module_outputs(self, outputs: tuple[Tensor, ...]) -> ModuleV3Outputs:
        """Names the outputs of :meth:`forward` for the pipeline model."""
        if not self.emit_last_token_logits:
            raise ValueError(
                "The pipeline model reads the last-token logits, which this"
                " model was configured to skip."
            )
        rest = iter(outputs)
        next_token_logits = next(rest)
        logits = logit_offsets = hidden_states = None
        if self.return_logits is ReturnLogits.VARIABLE:
            logits, logit_offsets = next(rest), next(rest)
        if self.return_hidden_states is ReturnHiddenStates.ALL_NORMALIZED:
            hidden_states = next(rest)
        return ModuleV3Outputs(
            next_token_logits=next_token_logits,
            logits=logits,
            logit_offsets=logit_offsets,
            hidden_states=hidden_states,
        )

    def _check_output_modes(self) -> None:
        supported_logits = (ReturnLogits.LAST_TOKEN, ReturnLogits.VARIABLE)
        supported_hidden = (
            ReturnHiddenStates.NONE,
            ReturnHiddenStates.ALL_NORMALIZED,
        )
        if (
            self.return_logits not in supported_logits
            or self.return_hidden_states not in supported_hidden
        ):
            raise NotImplementedError(
                f"return_logits={self.return_logits} with "
                f"return_hidden_states={self.return_hidden_states} is not "
                "implemented on ModuleV3."
            )
        if not self.emit_last_token_logits and (
            self.return_logits is not ReturnLogits.VARIABLE
        ):
            raise ValueError(
                "Skipping the last-token logits requires VARIABLE logits."
            )

    def _project(self, h: Tensor) -> Tensor:
        return project_to_logits(
            h,
            self.norm,
            self.lm_head,
            self.mesh,
            self.config.vocab_size,
            self.config.unpadded_vocab_size,
        )

    def _variable_logits(
        self, h: Tensor, input_row_offsets: Tensor, return_n_logits: Tensor
    ) -> tuple[Tensor, Tensor]:
        assert self.mesh is not None
        # Under data parallelism each replica holds only its own rows, and
        # the vocab-sharded head needs the whole batch on every device.
        variable_h, offsets = gather_variable_rows(
            h,
            input_row_offsets,
            return_n_logits,
            self.mesh,
            data_parallel_degree=self.config.data_parallel_degree,
        )
        return self._project(variable_h), offsets


def project_to_logits(
    h: Tensor,
    norm: Callable[[Tensor], Tensor],
    lm_head: Callable[[Tensor], Tensor],
    mesh: DeviceMesh | None,
    vocab_size: int,
    unpadded_vocab_size: int | None,
) -> Tensor:
    """Returns the float32 logits of ``h`` on the first device of ``mesh``.

    Vocabulary rows at or past ``unpadded_vocab_size`` are masked to negative
    infinity.

    On a ``(dp, tp)`` mesh ``h`` holds each replica's own rows: every replica
    projects them over its ``tp`` axis, and the first device copies in each
    replica's logits. No collective spans the ``dp`` axis, whose groups are
    not runs of consecutive devices.
    """
    logits = lm_head(norm(h))
    if mesh is not None and replica_size(mesh) > 1 and DP in mesh.axis_names:
        logits = F.allgather(logits, tensor_axis=logits.rank - 1, mesh_axis=TP)
        logits = _replica_rows_on_first_device(logits, mesh)
    elif mesh is not None:
        logits = logits.to(mesh.devices[0])
    return mask_padded_tail(
        logits.cast(DType.float32), vocab_size, unpadded_vocab_size
    )


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
        """Runs the model over its graph inputs."""
        mesh = self.config.mesh
        assert mesh is not None

        # Reconstruct inputs from variadic arguments.
        data_parallel_splits: Tensor | None = None
        input_row_offsets_i64: Tensor | None = None
        dp_degree = self.config.data_parallel_degree
        batch_context_lengths = variadic_args[:dp_degree]
        variadic_args = variadic_args[dp_degree:]
        if dp_degree > 1:
            batch_context_lengths_tensor = replica_context_lengths(
                batch_context_lengths, mesh
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
        outputs = self.language_model(
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
        return self.language_model.module_outputs(outputs)


class DeepseekV3_2Decoder(Module[[DecoderInputs], DecoderOutputs]):
    """DeepSeek-V3.2's sparse-MLA decoder, registered for the ``DECODER`` role.

    It runs tensor-parallel attention over a ``tp`` mesh, one attention
    replica per device over a ``dp`` mesh, or, under expert parallelism,
    tensor-parallel replicas over a ``(dp, tp)`` mesh, with its experts
    tensor-parallel or, under expert parallelism, spread across the devices.
    """

    graph_name: ClassVar[str] = "DeepseekV3_2"
    supports_data_parallel: ClassVar[bool] = True
    supports_multi_device_replicas: ClassVar[bool] = True

    def __init__(self, config: DeepseekV3_2Config) -> None:
        super().__init__()
        if config.mesh is None:
            raise ValueError(
                "DeepSeek-V3.2 needs a mesh: apply the runtime first"
            )
        self.ep_batch_manager = (
            None
            if config.ep_config is None
            else EPBatchManager(config.ep_config)
        )
        # Quantized checkpoints keep norms, biases and embeddings in bf16.
        dtype = (
            DType.bfloat16 if config.quant_config is not None else config.dtype
        )
        with default_dtype(dtype):
            self.language_model = DeepseekV3_2TextModel(
                config, self.ep_batch_manager
            )
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
            assert inputs.host_input_row_offsets is not None
            batch_context_length = replica_context_lengths(
                inputs.batch_context_lengths, mesh
            )
            data_parallel_splits = inputs.data_parallel_splits
            input_row_offsets_i64 = F.cast(
                inputs.host_input_row_offsets, DType.int64
            )
        else:
            batch_context_length = inputs.batch_context_lengths[0]

        kv_inputs = iter(x._graph_value for x in tree.leaves(inputs.kv))
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

        comm_buffers: EPCommBuffers | None = None
        if self.ep_batch_manager is not None:
            assert inputs.ep is not None
            comm_buffers = self.ep_batch_manager.comm_buffers(
                [x._graph_value for x in tree.leaves(inputs.ep)]
            )
        outputs = self.language_model(
            inputs.tokens,
            collection("mla"),
            collection("indexer"),
            inputs.return_n_logits,
            inputs.input_row_offsets,
            batch_context_length,
            data_parallel_splits,
            input_row_offsets_i64,
            comm_buffers,
        )
        return self.language_model.module_outputs(outputs)
