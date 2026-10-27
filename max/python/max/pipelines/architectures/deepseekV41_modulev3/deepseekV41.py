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
"""The DeepSeek-V4.1 text trunk, tensor or data parallel over one mesh."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from max import functional as F
from max import tree
from max.dtype import DType
from max.graph import DeviceRef, TensorValue, ops
from max.nn import Module
from max.nn.common_layers.embedding import VocabParallelEmbedding
from max.nn.common_layers.kv_cache import PagedCacheValues
from max.nn.common_layers.linear import ColumnParallelLinear
from max.nn.legacy.comm.ep import EPBatchManager, EPCommBuffers
from max.nn.legacy.data_parallelism import split_batch_replicated
from max.nn.legacy.kv_cache import KVCacheInputsPerDevice, MultiKVCacheParams
from max.nn.legacy.transformer import ReturnHiddenStates, ReturnLogits
from max.nn.sequential import ModuleList
from max.pipelines.lib import ModuleV3Outputs
from max.tensor import Tensor, default_device, default_dtype
from max.tensor.sharding import (
    DeviceMapping,
    DeviceMesh,
    Sharded,
    Unknown,
)

if TYPE_CHECKING:
    from transformers import AutoConfig

from .interface import V41Config, extra_input_names
from .layer_resolver import resolve
from .layers.attention import RaggedRows, absolute_positions, row_ids
from .layers.block import DecoderLayer, leaf_values, slot_constants
from .layers.engram import Engram
from .layers.mhc import mhc_final_collapse, mhc_identity_writeback, mhc_post
from .layers.norm import RMSNorm
from .layers.rope import RopeParams
from .model_config import DeepseekV41Config


def _split_replicated_batch(
    t: Tensor,
    input_row_offsets: Tensor,
    input_row_offsets_i64: Tensor,
    data_parallel_splits: Tensor,
    mapping: DeviceMapping,
) -> tuple[Tensor, Tensor]:
    """Splits a replicated ragged batch into DP shards (tokens, offsets).

    The DeepseekV3 ModuleV3 pattern: each rank cuts its requests' rows out
    of its own replicated copy, so no rank receives another's data.
    """
    devices = [DeviceRef.from_device(d) for d in mapping.mesh.devices]
    t_shards, offsets_shards = split_batch_replicated(
        devices,
        [TensorValue(shard) for shard in t.local_shards],
        [TensorValue(shard) for shard in input_row_offsets.local_shards],
        TensorValue(input_row_offsets_i64),
        TensorValue(data_parallel_splits),
    )
    return (
        Tensor.from_shard_values(t_shards, mapping),
        Tensor.from_shard_values(offsets_shards, mapping),
    )


def _slice_requests(
    t: Tensor,
    data_parallel_splits: Tensor,
    mapping: DeviceMapping,
    prefix: str,
) -> Tensor:
    """Slices a replicated per-request tensor into each rank's requests.

    ``t`` is ``[batch, ...]`` over the DP mesh; rank ``r`` keeps rows
    ``[splits[r], splits[r + 1])`` -- the same request ranges the token
    split uses, cut from its own replicated copy.
    """
    splits = TensorValue(data_parallel_splits)
    shards = []
    for r, shard in enumerate(t.local_shards):
        shards.append(
            ops.slice_tensor(
                TensorValue(shard),
                [
                    (
                        slice(splits[r], splits[r + 1]),
                        f"{prefix}_requests_r{r}",
                    ),
                    ...,
                ],
            )
        )
    return Tensor.from_shard_values(shards, mapping)


def _per_rank_ragged_rows(
    row_offsets: Tensor,
    cache_lengths: Tensor,
    tokens: Tensor,
    mapping: DeviceMapping,
) -> RaggedRows:
    """Ragged rows per rank, from its own rebased offsets and lengths.

    Each rank's cache-lengths shard is its own replica's record (the KV
    inputs are per-device), so the per-rank rows are the same
    single-device computation the tensor-parallel trunk runs once.
    """
    shards = [
        RaggedRows(
            offsets,
            *row_ids(offsets, shard.shape[0]),
            absolute_positions(offsets, lengths, shard.shape[0]),
            F.cast(offsets[1:] - offsets[:-1], DType.int32),
        )
        for offsets, lengths, shard in zip(
            row_offsets.local_shards,
            cache_lengths.local_shards,
            tokens.local_shards,
            strict=True,
        )
    ]
    return RaggedRows(
        offsets=Tensor.from_shard_values([s.offsets for s in shards], mapping),
        bid=Tensor.from_shard_values([s.bid for s in shards], mapping),
        local=Tensor.from_shard_values([s.local for s in shards], mapping),
        positions=Tensor.from_shard_values(
            [s.positions for s in shards], mapping
        ),
        lengths=Tensor.from_shard_values([s.lengths for s in shards], mapping),
    )


def _tie_shape(
    t: TensorValue, like: TensorValue, *, offset: int = 0
) -> TensorValue:
    """Rebinds ``t``'s shape onto ``like``'s, plus ``offset`` on axis 0.

    Each rank's sliced fields carry independently-created symbolic dims
    (per-slice labels), so the engram kernel's host-side shape checks --
    ``lookback_dead`` matching ``lookback``, the row offsets holding
    ``lookback``'s requests plus one -- see unequal dims where the values
    agree. The rebind asserts the tie per rank.
    """
    target = like.shape if offset == 0 else [like.shape[0] + offset]
    return ops.rebind(t, target)


_tie_shape_op = F.functional(_tie_shape)


def _dp_axis_name(mesh: DeviceMesh) -> str:
    """The single mesh axis's name, ``dp`` under data parallelism."""
    return mesh.axis_names[0]


def _host_total_rows(
    splits: TensorValue, row_bounds: TensorValue, *, last: int
) -> TensorValue:
    """The batch's total token rows as a host-resident 0-d int64 scalar.

    ``splits[last]`` is the global request count (the DP split pair's last
    boundary) and ``row_bounds`` holds the cumulative row offsets, so the
    entry there is the last request's end row: the total every rank's shard
    rows sum to, without a device read. Both inputs are CPU-resident, which
    is what makes the grouped-MoE tile-regime estimate host arithmetic the
    launch reads in place instead of a device-to-host materialization.
    """
    return row_bounds[splits[last]]


_host_total_rows_op = F.functional(_host_total_rows)


@dataclass
class Batch:
    """One forward's staged inputs, shared by every layer.

    Under data parallelism (``dp > 1``) the per-row and per-request fields
    hold each rank's own shard: its requests' token rows, rebased row
    offsets and request-sliced extras. ``comm`` carries the EP communication
    buffers the MoE dispatch/combine reads, and the ``tokens_full`` /
    ``row_offsets_full`` pair keeps the pre-split replicated batch the
    vocab-parallel embedding reads before the split.
    """

    tokens: Tensor
    rows: RaggedRows
    swa_begin: Tensor
    global_write_floor: Tensor
    image_token_mask: Tensor
    lookback: Tensor
    lookback_dead: Tensor
    tables: list[Tensor]
    kv: dict[str, PagedCacheValues]
    comm: EPCommBuffers | None = None
    tokens_full: Tensor | None = None
    row_offsets_full: Tensor | None = None
    data_parallel_splits: Tensor | None = None
    input_row_offsets_i64: Tensor | None = None


class DeepseekV41(Module[..., ModuleV3Outputs]):
    """Runs all live rows through the text decoder, over one device mesh.

    With ``data_parallel_degree > 1`` the mesh axis is a data-parallel
    axis: every rank runs the full attention heads on its own request shard
    (the token split happens in-graph from ``data_parallel_splits``), the
    routed experts stay expert-sharded with a P2P dispatch/combine, and no
    residual-path all-reduce runs anywhere.
    """

    def __init__(
        self,
        config: DeepseekV41Config,
        kv_params: MultiKVCacheParams,
        mesh: DeviceMesh,
        *,
        text_config: AutoConfig,
        token_map: np.ndarray,
        data_parallel_degree: int = 1,
        ep_batch_manager: EPBatchManager | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.kv_params = kv_params
        self.mesh = mesh
        self.dp = data_parallel_degree
        self.ep_batch_manager = ep_batch_manager
        if self.dp > 1 and mesh.num_devices != self.dp:
            raise ValueError(
                f"data_parallel_degree {self.dp} must equal the mesh size "
                f"({mesh.num_devices}); DP attention runs TP=1 per rank."
            )
        if self.dp > 1 and ep_batch_manager is None:
            # The routed experts shard over the same DP mesh, so a
            # tensor-parallel MoE would reduce across ranks that hold
            # different requests. DP attention is only wired with the
            # expert-parallel MoE.
            raise ValueError(
                "data-parallel attention requires the expert-parallel MoE: "
                "pass the EP batch manager (ep_size must equal the device "
                "count)."
            )
        if self.dp > 1 and (
            config.return_logits != ReturnLogits.LAST_TOKEN
            or config.return_hidden_states != ReturnHiddenStates.NONE
        ):
            # The all-row and variable logit paths (and the hidden-state
            # paths that read the same gathers) are wired for the
            # replicated rows of tensor parallelism; under DP they would
            # silently cover one rank's shard only.
            raise ValueError(
                "data parallelism serves next-token logits only; the "
                "all-row and variable logit outputs and the hidden-state "
                "outputs stay tensor-parallel."
            )
        self.interface_config = V41Config.from_hf(
            {"text_config": vars(text_config)}
        )
        rope = RopeParams.from_text_config(text_config)
        with default_device(mesh), default_dtype(DType.bfloat16):
            # The vocab-sharded embedding and head split their weight over
            # the mesh axis by name; under DP that axis is ``dp``.
            self.embed = VocabParallelEmbedding(
                config.vocab_size,
                dim=config.hidden_size,
                tp_axis=_dp_axis_name(mesh) if self.dp > 1 else None,
            )
            layers = []
            for info in resolve(config):
                engram = None
                if info.layer_id in config.engram_layer_ids:
                    engram = Engram(
                        layer_ids=tuple(config.engram_layer_ids),
                        num_embeddings=tuple(config.engram_num_embeddings),
                        max_ngram_size=int(text_config.engram_max_ngram_size),
                        n_heads=int(text_config.engram_n_heads),
                        head_dim=int(text_config.engram_head_dim),
                        compressed_vocab_size=int(
                            text_config.engram_compressed_vocab_size
                        ),
                        engram_vocab_size=int(text_config.engram_vocab_size),
                        hidden_size=config.hidden_size,
                        hc_mult=config.hc_mult,
                        pad_token_id=int(text_config.engram_pad_token_id),
                        rms_norm_eps=config.rms_norm_eps,
                        device=mesh,
                        token_map=token_map,
                    )
                layers.append(
                    DecoderLayer(
                        config,
                        info,
                        mesh,
                        rope=rope,
                        max_position=config.max_seq_len,
                        candidate_block_size=config.candidate_block_size,
                        engram=engram,
                        ep_batch_manager=ep_batch_manager,
                        data_parallel=self.dp > 1,
                    )
                )
            self.layers = ModuleList(layers)
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.head = ColumnParallelLinear(
                config.hidden_size,
                config.vocab_size,
                bias=False,
                tp_axis=_dp_axis_name(mesh) if self.dp > 1 else None,
            )

    def batch_from_parts(
        self,
        tokens: Tensor,
        input_row_offsets: Tensor,
        kv: Mapping[str, PagedCacheValues],
        extras: Mapping[str, Tensor],
        *,
        data_parallel_splits: Tensor | None = None,
        input_row_offsets_i64: Tensor | None = None,
        comm_buffers: EPCommBuffers | None = None,
    ) -> Batch:
        """Stages already-split inputs onto the mesh, as a :class:`Batch`.

        ``forward`` unflattens the graph's variadic tail into per-leaf
        caches and named extras and lands here; a spec-decode target
        adapter that starts from the driver's per-leaf caches can call
        this directly, so the two paths cannot drift apart.

        Under data parallelism (``self.dp > 1``) the DP split inputs are
        required and every per-row/per-request field comes out per rank.

        Raises:
            ValueError: If ``self.dp > 1`` and either half of the DP split
                pair is missing.
        """
        mesh = self.mesh
        tokens = F.distributed_broadcast(tokens, mesh)
        offsets = F.distributed_broadcast(input_row_offsets, mesh)
        swa_begin = F.distributed_broadcast(extras["swa_begin"], mesh)
        floor = F.distributed_broadcast(extras["global_write_floor"], mesh)
        image_token_mask = F.distributed_broadcast(
            extras["image_token_mask"], mesh
        )
        lookback = F.distributed_broadcast(extras["engram_lookback"], mesh)
        lookback_dead = F.distributed_broadcast(
            extras["engram_lookback_dead"], mesh
        )
        tables = [
            extras[f"engram_table_{i}_{part}"]
            for i in range(len(self.config.engram_layer_ids))
            for part in ("weight", "scale")
        ]
        if self.dp > 1:
            if data_parallel_splits is None or input_row_offsets_i64 is None:
                raise ValueError(
                    "data parallelism requires the data_parallel_splits and "
                    "input_row_offsets_i64 graph inputs."
                )
            shard_mapping = DeviceMapping(mesh, (Sharded(0),))
            tokens_full, offsets_full = tokens, offsets
            tokens, offsets = _split_replicated_batch(
                tokens,
                offsets,
                input_row_offsets_i64,
                data_parallel_splits,
                shard_mapping,
            )
            # The image mask is per step token (the router reads it per
            # row), so it splits with the tokens.
            image_token_mask, _ = _split_replicated_batch(
                image_token_mask,
                offsets_full,
                input_row_offsets_i64,
                data_parallel_splits,
                shard_mapping,
            )
            swa_begin = _slice_requests(
                swa_begin, data_parallel_splits, shard_mapping, "swa_begin"
            )
            floor = _slice_requests(
                floor, data_parallel_splits, shard_mapping, "write_floor"
            )
            lookback = _slice_requests(
                lookback, data_parallel_splits, shard_mapping, "lookback"
            )
            lookback_dead = _slice_requests(
                lookback_dead,
                data_parallel_splits,
                shard_mapping,
                "lookback_dead",
            )
            # The engram kernel's shape checks compare symbolic dims, and
            # each sliced field carries its own; tie the companion's dims
            # onto ``lookback``'s, per rank (the rule-less tie drops the
            # placement, so re-claim the row sharding).
            lookback_dead = _tie_shape_op(
                lookback_dead, lookback
            ).rebind_mapping(lookback.mapping)
            rows = _per_rank_ragged_rows(
                offsets,
                kv["swa"].cache_lengths,
                tokens,
                shard_mapping,
            )
        else:
            positions = absolute_positions(
                offsets, kv["swa"].cache_lengths, tokens.shape[0]
            )
            bid, local = row_ids(offsets, tokens.shape[0])
            rows = RaggedRows(
                offsets,
                bid,
                local,
                positions,
                F.cast(offsets[1:] - offsets[:-1], DType.int32),
            )
        return Batch(
            tokens=tokens,
            rows=rows,
            swa_begin=swa_begin,
            global_write_floor=floor,
            image_token_mask=image_token_mask,
            lookback=lookback,
            lookback_dead=lookback_dead,
            tables=tables,
            kv=dict(kv),
            comm=comm_buffers,
            tokens_full=(tokens_full if self.dp > 1 else None),
            row_offsets_full=(offsets_full if self.dp > 1 else None),
            data_parallel_splits=data_parallel_splits,
            input_row_offsets_i64=input_row_offsets_i64,
        )

    def embed_stream(self, b: Batch) -> tuple[Tensor, Tensor]:
        """The embedding on ``hc_mult`` copies and the one-hot first ``pre``.

        Under DP the embedding runs on the full replicated batch (the
        vocab-sharded gather all-reduces while every rank still holds the
        same rows) and the stream and ``pre`` are split afterwards, so the
        per-rank trunk starts from its own rows.
        """
        hc = self.config.hc_mult
        full = b.tokens_full if b.tokens_full is not None else b.tokens
        h = self.embed(full)
        stream = F.stack([h] * hc, axis=1)
        rows = h.shape[0]
        pre = F.concat(
            [
                F.full([rows, 1], 1.0, dtype=DType.float32, device=h.mapping),
                F.full(
                    [rows, hc - 1], 0.0, dtype=DType.float32, device=h.mapping
                ),
            ],
            axis=1,
        )
        if b.tokens_full is not None:
            assert (
                b.row_offsets_full is not None
                and b.data_parallel_splits is not None
                and b.input_row_offsets_i64 is not None
                and self.dp > 1
            )
            mapping = DeviceMapping(self.mesh, (Sharded(0),))
            stream, _ = _split_replicated_batch(
                stream,
                b.row_offsets_full,
                b.input_row_offsets_i64,
                b.data_parallel_splits,
                mapping,
            )
            pre, _ = _split_replicated_batch(
                pre,
                b.row_offsets_full,
                b.input_row_offsets_i64,
                b.data_parallel_splits,
                mapping,
            )
        return stream, pre

    def _run_trunk(
        self, b: Batch, stream: Tensor, pre: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Every layer over the staged batch.

        Returns:
            The written-back stream with its carried ``pre``.
        """
        rows = b.rows
        offsets = rows.offsets
        positions = rows.positions
        # The grouped MoE's tile-regime estimate: the batch's total token
        # rows read off the CPU DP pair, so the launch needs no
        # device-to-host materialization. ``None`` without data parallelism.
        total_rows = (
            None
            if b.data_parallel_splits is None or b.input_row_offsets_i64 is None
            else _host_total_rows_op(
                b.data_parallel_splits, b.input_row_offsets_i64, last=self.dp
            )
        )
        y, post, comb = mhc_identity_writeback(stream)
        hc = self.config.hc_mult
        selection: dict[str, Tensor] = {}
        for layer in self.layers:
            if layer.layer_id in self.config.engram_layer_ids:
                stream = mhc_post(
                    y, stream, post, comb.reshape([stream.shape[0], hc, hc]), hc
                )
                ordinal = F.constant(
                    self.config.engram_layer_ids.index(layer.layer_id),
                    DType.uint32,
                    device=self.mesh.devices[0],
                )
                ordinal = F.distributed_broadcast(ordinal, self.mesh)
                if self.dp > 1:
                    # The engram kernel wants the offsets' length tied to
                    # the batch it shares with ``lookback``. Under DP each
                    # rank's sliced fields carry their own dims, so the
                    # tie is per rank -- a global-shape rebind would
                    # misread the row-sharded offsets -- and the rule-less
                    # tie drops the placement, so re-claim the row
                    # sharding.
                    tied_offsets = _tie_shape_op(
                        offsets, b.lookback, offset=1
                    ).rebind_mapping(b.lookback.mapping)
                else:
                    tied_offsets = F.rebind(
                        offsets,
                        [b.lookback.shape[0] + 1],
                        "Engram offsets contain batch+1 entries",
                    )
                (stream,) = layer.engram(
                    [],
                    b.tables,
                    [ordinal, b.lookback, b.lookback_dead, b.image_token_mask],
                    hidden=stream,
                    tokens=b.tokens,
                    positions=positions.cast(DType.int64),
                    query_start_loc=tied_offsets,
                )
                y, post, comb = mhc_identity_writeback(stream)
            routed = {
                "reuse_r2": selection.get("top_k_r2"),
                "reindex_r1": selection.get("pool"),
                "reuse_r1": selection.get("top_k_r1"),
            }.get(layer.group)
            stream, pre, y, post, comb, produced = layer(
                stream,
                pre,
                y,
                post,
                comb,
                rows,
                b.swa_begin,
                b.global_write_floor,
                b.image_token_mask,
                leaf_values(layer.plan, b.kv),
                slot_constants(layer.plan),
                [] if routed is None else [routed],
                b.comm,
                total_rows,
            )
            if layer.group == "full_r2":
                selection["top_k_r2"] = produced[0]
            elif layer.group == "full_r1_cand":
                selection["top_k_r1"], selection["pool"] = produced[:2]
            elif layer.group == "reindex_r1":
                selection["top_k_r1"] = produced[0]
        stream = mhc_post(
            y, stream, post, comb.reshape([stream.shape[0], hc, hc]), hc
        )
        return stream, pre

    @staticmethod
    def _gather_last_tokens(h: Tensor, row_offsets: Tensor) -> Tensor:
        """Gathers each request's last row from the ragged batch, per shard.

        Under data parallelism ``row_offsets`` is rebased into each rank's
        local frame, so its values only address that rank's own rows; the
        gather runs on each device's own rows, never through an
        auto-sharded ``F.gather`` (the DeepseekV3 ModuleV3 pattern).
        """

        def _body(hv: TensorValue, offsets: TensorValue) -> TensorValue:
            return ops.gather(hv, offsets[1:] - 1, axis=0)

        if not h.is_distributed:
            return F.gather(h, row_offsets[1:] - 1, axis=0)
        return F.functional(_body)(h, row_offsets).rebind_mapping(h.mapping)

    def forward(
        self,
        tokens: Tensor,
        return_n_logits: Tensor,
        input_row_offsets: Tensor,
        *rest: Tensor,
    ) -> ModuleV3Outputs:
        data_parallel_splits: Tensor | None = None
        input_row_offsets_i64: Tensor | None = None
        if self.dp > 1:
            # The compile ABI puts the DP split pair right after the row
            # offsets, before the KV leaves (the batch processor inserts
            # them at the same positions).
            data_parallel_splits = rest[0]
            input_row_offsets_i64 = rest[1]
            rest = rest[2:]
        n_kv = len(self.kv_params.flattened_kv_inputs())
        upstream = self.kv_params.unflatten_kv_inputs(
            iter(x._graph_value for x in rest[:n_kv])
        )
        # Replicas serve different requests, so their caches vary; under
        # tensor parallelism the devices all store the same latent cache.
        kv_mapping = (
            DeviceMapping(self.mesh, (Unknown(),))
            if self.dp > 1
            else DeviceMapping.replicated(self.mesh)
        )
        caches = {
            name: PagedCacheValues.from_upstream(
                tree.leaves(upstream[name], leaf=KVCacheInputsPerDevice),
                kv_mapping,
            )
            for name in self.kv_params.children
        }
        names = extra_input_names(self.interface_config)
        consumed = n_kv + len(names)
        extras = dict(zip(names, rest[n_kv:consumed], strict=True))
        comm_buffers: EPCommBuffers | None = None
        ep_rest = list(rest[consumed:])
        if self.ep_batch_manager is not None:
            # Whatever trails the extras is the EP tail the compile input
            # list appended (the batch manager's own input types).
            comm_buffers = self.ep_batch_manager.comm_buffers(ep_rest)
        elif ep_rest:
            raise ValueError(
                f"{len(ep_rest)} inputs after the extras, but the trunk "
                "holds no EP batch manager to bind them."
            )
        b = self.batch_from_parts(
            tokens,
            input_row_offsets,
            caches,
            extras,
            data_parallel_splits=data_parallel_splits,
            input_row_offsets_i64=input_row_offsets_i64,
            comm_buffers=comm_buffers,
        )
        stream, pre = self.embed_stream(b)
        stream, pre = self._run_trunk(b, stream, pre)
        hidden = mhc_final_collapse(stream, pre)
        return self._outputs(hidden, b.rows.offsets, return_n_logits)

    def _outputs(
        self, hidden: Tensor, offsets: Tensor, return_n_logits: Tensor
    ) -> ModuleV3Outputs:
        if self.dp > 1:
            # One request's last row lives on its own rank (the offsets are
            # rebased per rank), so the gather runs per shard and one
            # allgather -- in rank order, which is global request order by
            # construction of the splits -- hands the head every
            # replica's requests.
            normalized = self.norm(hidden)
            last = F.allgather(
                self._gather_last_tokens(normalized, offsets), tensor_axis=0
            )
            return ModuleV3Outputs(next_token_logits=self._compute_logits(last))
        last_raw = F.gather(hidden, offsets[1:] - 1, axis=0)
        hidden = self.norm(hidden)
        last = F.gather(hidden, offsets[1:] - 1, axis=0)
        next_logits = self._compute_logits(last)
        logits = None
        logit_offsets = None
        if self.config.return_logits == ReturnLogits.ALL:
            logits = self._compute_logits(hidden)
            logit_offsets = (
                offsets.local_shards[0] if offsets.is_distributed else offsets
            )
        elif self.config.return_logits == ReturnLogits.VARIABLE:
            count = return_n_logits[0]
            suffix = F.range(
                count,
                0,
                -1,
                out_dim="return_n_logits_range",
                device=hidden.mapping,
                dtype=DType.int64,
            )
            indices = (offsets[1:].unsqueeze(-1) - suffix).reshape([-1])
            logits = self._compute_logits(F.gather(hidden, indices, axis=0))
            logit_offsets = F.range(
                0,
                indices.shape[0] + count,
                count,
                out_dim="logit_offsets",
                device=hidden.mapping,
                dtype=DType.int64,
            )
            if logit_offsets.is_distributed:
                logit_offsets = logit_offsets.local_shards[0]
        hidden_states = None
        if self.config.return_hidden_states == ReturnHiddenStates.LAST:
            # Pre-final-norm rows, the same contract as the Llama3
            # ModuleV3 model's ``last_h``; the head consumes the
            # normalized ones.
            hidden_states = (
                last_raw.local_shards[0]
                if last_raw.is_distributed
                else last_raw
            )
        elif (
            self.config.return_hidden_states
            == ReturnHiddenStates.ALL_NORMALIZED
        ):
            hidden_states = (
                hidden.local_shards[0] if hidden.is_distributed else hidden
            )
        return ModuleV3Outputs(
            next_token_logits=next_logits,
            logits=logits,
            logit_offsets=logit_offsets,
            hidden_states=hidden_states,
        )

    def _compute_logits(self, hidden: Tensor) -> Tensor:
        logits = self.head(hidden)
        if logits.is_distributed:
            logits = F.allgather(logits, tensor_axis=-1).local_shards[0]
        return logits.cast(DType.float32)
