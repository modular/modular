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
from max.graph import BufferValue, TensorValue
from max.nn import Module, as_subgraph
from max.nn.common_layers.embedding import VocabParallelEmbedding
from max.nn.common_layers.kv_cache import PagedCacheValues, kv_input_values
from max.nn.common_layers.linear import ColumnParallelLinear
from max.nn.common_layers.mesh_axis import DP, TP
from max.nn.common_layers.multi_latent_attention import MLAPrefillMetadata
from max.nn.common_layers.quant_moe import QuantizedMoE
from max.nn.common_layers.rotary_embedding import RotaryEmbedding
from max.nn.ep import EPBuffers
from max.nn.ep import init as ep_init
from max.nn.ep_config import EPConfig
from max.nn.kernels import mla_sparse_selection_rows
from max.nn.kv_cache import KVCacheInputsPerDevice, KVCacheParamInterface
from max.nn.outputs import ReturnHiddenStates, ReturnLogits
from max.nn.rope_scaling import DeepseekYarnRopeScalingParams
from max.pipelines.lib import ModuleV3Outputs
from max.pipelines.lib.role_contracts import DecoderInputs, DecoderOutputs
from max.pipelines.lib.utils import split_replicated_batch
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
)
from .layers.prefill_context_parallel import (
    PrefillContextParallelStep,
    prefill_predicate,
)
from .layers.rms_norm import MultiplyBeforeCastRMSNorm
from .layers.sparse_mla import CarriedSelection, Selection
from .layers.transformer_block import (
    BlockOutput,
    DeepseekV3_2TransformerBlock,
)
from .model_config import (
    DeepseekV3_2Config,
    HiSparseInputs,
    hisparse_gather_groups,
    indexer_cache_layer_map,
    stage_indexer_types,
)


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
    first = mesh.devices[0]
    rows = [
        t.local_shards[r * group].to(first)
        for r in range(mesh.num_devices // group)
    ]
    return F.concat(rows, axis=0)


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
        [batch_context_lengths[i // group] for i in range(mesh.num_devices)],
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


class DeepseekV3_2TextModel(Module[..., tuple[Tensor, ...]]):
    """The DeepseekV3.2 language model.

    DeepseekV3 with sparse attention: a lightning indexer selects the keys each
    query attends to, backed by its own FP8 key cache.
    """

    def __init__(
        self,
        config: DeepseekV3_2Config,
        ep_config: EPConfig | None = None,
    ) -> None:
        self.ep_config = ep_config

        self.rope = rotary_embedding(config)

        # One device per replica shards the vocabulary over the ``dp`` axis;
        # replicas of several devices shard it over their ``tp`` axis.
        tp_axis = (
            DP
            if config.data_parallel_degree > 1
            and (config.mesh is None or TP not in config.mesh.axis_names)
            else None
        )
        # A pipeline stage builds only its layers, under their global names;
        # only the first embeds and only the last scores. A last stage whose
        # NextN layer drafts also embeds, for the draft.
        ids = (
            range(config.num_hidden_layers)
            if config.layer_ids is None
            else config.layer_ids
        )
        self.first = ids.start == 0
        self.last = ids.stop == config.num_hidden_layers
        if self.first or (self.last and config.last_stage_embeds):
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                dim=config.hidden_size,
                tp_axis=tp_axis,
            )
        if self.last:
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
        self.dim = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.layers = {
            i: DeepseekV3_2TransformerBlock(
                config=config,
                layer_idx=i,
                attention_scale=scale,
                ep_config=ep_config
                if i >= config.first_k_dense_replace
                else None,
            )
            for i in ids
        }
        # A stage before the last hands its last selection to the next stage,
        # as token positions.
        if not self.last:
            for layer in self.layers.values():
                layer.self_attn.carries_positions = True
        # The caches hold only this stage's layers, so its indexer slots and
        # gather widths count from its first layer.
        self.indexer_types = stage_indexer_types(config.indexer_types, ids)
        self.kv_params = config.kv_params
        self.config = config
        self.mesh = config.mesh
        # A unified spec-decode graph sets these as the graph-API model does:
        # it reads VARIABLE logits and every token's normalized hidden state,
        # and skips the last-token projection it never uses.
        self.return_logits = ReturnLogits.LAST_TOKEN
        self.return_hidden_states = ReturnHiddenStates.NONE
        self.emit_last_token_logits = True

    @property
    def first_layer(self) -> DeepseekV3_2TransformerBlock:
        """The first block this model builds."""
        return next(iter(self.layers.values()))

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
        ep: EPBuffers | None = None,
        hisparse: HiSparseInputs | None = None,
        boundary: Sequence[Tensor] | None = None,
    ) -> tuple[Tensor, ...]:
        """Runs the embedding and decoder stack over the tokens.

        A pipeline stage after the first starts from ``boundary``, each
        device's hidden states from the previous stage. A stage before the
        last returns its own, in the same form, in place of the logits.
        """
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

        carried: Tensor | None = None
        if self.first:
            h = self.embed_tokens(tokens)
        else:
            # Each device's hidden states, then its last top-k selection's
            # token positions, which a stage that starts on a ``shared``
            # layer reuses.
            assert boundary is not None and self.mesh is not None
            replicated = DeviceMapping(
                self.mesh, tuple(Replicated() for _ in self.mesh.axis_names)
            )
            n = self.mesh.num_devices
            h = Tensor.from_shard_values(boundary[:n], replicated)
            carried = Tensor.from_shard_values(boundary[n:], replicated)
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
        first_attn = self.first_layer.self_attn
        if first_attn.graph_mode in ("prefill", "auto"):
            mla_prefill_metadata = first_attn.create_mla_prefill_metadata(
                input_row_offsets, kv_collection
            )
            # Host-substitute the per-layer D2H buffer_length copies with the
            # CPU batch_context_length so the graph stays capturable.
            mla_prefill_metadata.buffer_lengths = batch_context_length

        layer_args = (
            kv_collection,
            indexer_kv_collection,
            input_row_offsets,
            freqs_cis,
            mla_prefill_metadata,
            ep,
        )
        pcp: PrefillContextParallelStep | None = None
        if self.config.prefill_context_parallel:
            # Prefill context parallelism is a per-step choice: a prefill
            # splits its rows over each DCP group, while decode and verify
            # steps keep DCP. The step's rows are predicated: each layer's
            # attention runs PCP on a PCP step's rows and DCP on any other's.
            assert self.mesh is not None
            first_attn = self.first_layer.self_attn
            active = prefill_predicate(
                tokens, input_row_offsets, batch_context_length
            )
            # One plan for every layer, built on every step: a step that
            # does not run PCP sizes it to one page and one row.
            pcp = PrefillContextParallelStep(
                dcp=first_attn.dcp_size,
                interleave=first_attn.dcp_interleave,
                page_size=self.config.mla_kv_params.page_size,
                namespace="pcp",
                active=active,
                batch_context_length=batch_context_length,
            ).predicated(h)
            # Predicated: both paths run every step, each on its rows.
            pcp = pcp.with_plan(
                input_row_offsets,
                pcp.rows_split(h)[0],
                kv_collection,
                indexer_kv_collection,
            )
        h, selection = self._run_layers(
            h, *layer_args, pcp=pcp, hisparse=hisparse, carried=carried
        )
        if not self.last:
            # The next stage's first block applies its own input norm.
            assert isinstance(selection, CarriedSelection)
            return (*h.local_shards, *selection.positions.local_shards)

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

    def _run_layers(
        self,
        h: Tensor,
        kv_collection: PagedCacheValues,
        indexer_kv_collection: PagedCacheValues,
        input_row_offsets: Tensor,
        freqs_cis: Tensor,
        mla_prefill_metadata: MLAPrefillMetadata | None,
        ep: EPBuffers | None,
        pcp: PrefillContextParallelStep | None = None,
        hisparse: HiSparseInputs | None = None,
        carried: Tensor | None = None,
    ) -> tuple[Tensor, Selection | None]:
        """Runs the decoder blocks over ``h``.

        With ``pcp`` every block's attention chooses prefill context
        parallelism per step. With ``hisparse`` every ``full`` layer resolves
        its selection against the hot buffer and gathers the misses. A
        ``shared`` first layer reuses ``carried``, the token positions of the
        previous stage's last selection, resolved against this stage's cache.
        Returns the hidden states and the last selection.
        """
        # MoE blocks share a subgraph, as in V3. ``full`` and ``shared``
        # layers differ structurally (shared layers carry no indexer weights),
        # so they get one group each: a shared subgraph name is what makes two
        # call sites reuse a body, and the cache key does not otherwise see
        # the difference.
        topk_indices: Selection | None = None
        # A ``full`` first layer selects for itself, so only a ``shared`` one
        # reads the carried positions.
        if carried is not None and self.indexer_types[0] == "shared":
            rows = mla_sparse_selection_rows(
                carried, input_row_offsets, kv_collection
            )
            topk_indices = (
                rows if self.last else CarriedSelection(rows, carried)
            )
        layers = list(self.layers.values())
        indexer_slots = indexer_cache_layer_map(self.indexer_types, len(layers))
        # How many consecutive layers one gather launch fills at each layer:
        # the full layer and the ``shared`` layers that reuse its top-k.
        gather_groups = hisparse_gather_groups(self.indexer_types, len(layers))
        # Each block hands the next its input norm, fused into its post-MLP
        # all-gather, as the graph-API model does. Only a stage's input is
        # normalized here.
        carry = self.first_layer.carry_next_input_norm
        norm_h = self.first_layer.input_layernorm(h) if carry else None
        last = len(layers) - 1
        # ``idx`` is the layer's slot in this stage's cache.
        for idx, layer in enumerate(layers):
            layer_idx_tensor = F.constant(idx, DType.uint32, device=CPU())
            indexer_slot_tensor = F.constant(
                indexer_slots[idx], DType.uint32, device=CPU()
            )
            # A runtime input, not a Python constant: the ``full`` layers
            # share one subgraph body but not one gather width.
            gather_group = (
                F.constant(gather_groups[idx], DType.uint32, device=CPU())
                if hisparse is not None
                else None
            )
            call: Callable[..., BlockOutput] = layer
            group = "shared" if layer.self_attn.skip_topk else "full"
            if isinstance(layer.mlp, QuantizedMoE):
                call = as_subgraph(layer, name=f"moe_block_{group}")
            elif self.config.decode_context_parallel_degree > 1:
                # The memory planner gives the top-level graph one pool that
                # lives for the whole forward, and an inline block's
                # transients land in it. Under DCP every rank attends all
                # heads, so a dense block's peak is ~14 GiB at a 32K-token
                # prefill: inline, that is held through all 78 layers on top
                # of each MoE block's own pool, and the step no longer fits.
                # As a subgraph call it is planned and freed per call. The
                # name is the body's identity, so it carries the indexer
                # group as the MoE blocks' does.
                call = as_subgraph(layer, name=f"dense_block_{group}")
            # A block takes the next block's weight as an argument, since
            # a shared subgraph resolves only its own weights.
            next_input_gamma = (
                layers[idx + 1].input_layernorm.weight.cast(h.dtype)
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
                ep,
                topk_indices,
                False,
                indexer_slot_tensor,
                norm_x=norm_h,
                next_input_gamma=next_input_gamma,
                pcp=pcp,
                hisparse=hisparse,
                hisparse_group_size=gather_group,
            )
            norm_h = rest[0] if rest else None
        return h, topk_indices

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


def hisparse_inputs_from_values(
    config: DeepseekV3_2Config,
    values: Sequence[TensorValue | BufferValue | Tensor],
) -> HiSparseInputs | None:
    """Groups the HiSparse graph inputs into the tables one layer reads.

    The host and hot tables arrive one per device, like the KV leaves, and
    are stacked the same way so each layer reads its own device's shard;
    the host mirror is the one CPU tensor. Returns ``None``, and refuses
    any trailing input, when the graph was built without HiSparse.

    Args:
        config: The config the graph was built from, which sizes the tables.
        values: The HiSparse graph inputs, in their declared order: the
            host tables, the hot tables, then the host mirror. Graph values
            or single-device tensors, as the plain graph and the fused
            spec graph hand them over.

    Raises:
        ValueError: If the values disagree with what the config declares.
    """
    if config.hisparse is None:
        if values:
            raise ValueError(
                "the graph takes trailing inputs past the EP buffers, but"
                " hisparse is not set"
            )
        return None
    n_devices = len(config.devices)
    if len(values) != 2 * n_devices + 1:
        raise ValueError(
            "hisparse is set, so the graph takes one host table and one"
            " hot table per device plus the host mirror past the EP"
            f" buffers ({2 * n_devices + 1} inputs); got {len(values)}"
        )
    host_tables, hot_tables, host_mirror = (
        values[:n_devices],
        values[n_devices : 2 * n_devices],
        values[-1],
    )
    mesh = config.mesh
    assert mesh is not None
    # Mapped like the KV leaves: tensor-parallel devices stage the same
    # tables, and data-parallel replicas each stage their own rows'.
    mapping = DeviceMapping(
        mesh,
        tuple(
            Unknown() if name == DP else Replicated()
            for name in mesh.axis_names
        ),
    )
    return HiSparseInputs(
        host_table=Tensor.from_shard_values(list(host_tables), mapping),
        hot_table=Tensor.from_shard_values(list(hot_tables), mapping),
        host_mirror=Tensor.from_shard_values([host_mirror]),
        params=config.hisparse,
    )


class DeepseekV3_2(Module[..., ModuleV3Outputs]):
    """Top-level DeepseekV3.2 wrapper that unflattens variadic KV cache args."""

    def __init__(
        self,
        config: DeepseekV3_2Config,
        kv_params: KVCacheParamInterface,
        ep_config: EPConfig | None = None,
    ) -> None:
        super().__init__()
        self.language_model = DeepseekV3_2TextModel(config, ep_config)
        self.config = config
        self.kv_params = kv_params

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

        kv_inputs = iter(kv_input_values(variadic_args))
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

        # The EP inputs sit between the KV cache and the HiSparse tail.
        ep = ep_init.flat_buffers(
            self.language_model.ep_config, kv_inputs, mesh
        )
        hisparse = self._hisparse_inputs(list(kv_inputs))
        outputs = self.language_model(
            tokens,
            kv_collection,
            indexer_kv_collection,
            return_n_logits,
            input_row_offsets,
            batch_context_lengths_tensor,
            data_parallel_splits,
            input_row_offsets_i64,
            ep,
            hisparse,
        )
        return self.language_model.module_outputs(outputs)

    def _hisparse_inputs(
        self, values: Sequence[TensorValue | BufferValue]
    ) -> HiSparseInputs | None:
        """Groups the trailing graph inputs into the HiSparse tables.

        The host and hot tables arrive one per device, like the KV leaves, and
        are stacked the same way so each layer reads its own device's shard;
        the host mirror is the one CPU tensor. Returns ``None``, and refuses
        any trailing input, when the graph was built without HiSparse.
        """
        return hisparse_inputs_from_values(self.config, values)


class DeepseekV3_2Decoder(
    Module[[DecoderInputs], DecoderOutputs | tuple[Tensor, ...]]
):
    """DeepSeek-V3.2's sparse-MLA decoder, registered for the ``DECODER`` role.

    It runs tensor-parallel attention over a ``tp`` mesh, one attention
    replica per device over a ``dp`` mesh, or, under expert parallelism,
    tensor-parallel replicas over a ``(dp, tp)`` mesh, with its experts
    tensor-parallel or, under expert parallelism, spread across the devices.
    """

    graph_name: ClassVar[str] = "DeepseekV3_2"
    supports_data_parallel: ClassVar[bool] = True
    supports_multi_device_replicas: ClassVar[bool] = True
    supports_pipeline_stages: ClassVar[bool] = True

    def __init__(self, config: DeepseekV3_2Config) -> None:
        super().__init__()
        if config.mesh is None:
            raise ValueError(
                "DeepSeek-V3.2 needs a mesh: apply the runtime first"
            )
        # Quantized checkpoints keep norms, biases and embeddings in bf16.
        dtype = (
            DType.bfloat16 if config.quant_config is not None else config.dtype
        )
        with default_dtype(dtype):
            self.language_model = DeepseekV3_2TextModel(
                config, config.ep_config
            )
        self.config = config
        self.kv_params = config.kv_params

    def forward(
        self, inputs: DecoderInputs
    ) -> DecoderOutputs | tuple[Tensor, ...]:
        """Runs the decoder over the batch.

        A pipeline stage before the last returns each device's hidden
        states, which the next stage takes as its stage boundary.
        """
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

        kv_inputs = iter(kv_input_values(inputs.kv))
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

        outputs = self.language_model(
            inputs.tokens,
            collection("mla"),
            collection("indexer"),
            inputs.return_n_logits,
            inputs.input_row_offsets,
            batch_context_length,
            data_parallel_splits,
            input_row_offsets_i64,
            inputs.ep,
            boundary=(
                None
                if inputs.stage_boundary is None
                else list(inputs.stage_boundary)
            ),
        )
        if not self.language_model.last:
            return outputs
        return self.language_model.module_outputs(outputs)
