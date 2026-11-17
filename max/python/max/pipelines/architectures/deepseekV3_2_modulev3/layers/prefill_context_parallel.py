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
"""Prefill context parallelism (PCP) on top of decode context parallelism.

Under decode context parallelism (DCP, :mod:`max.nn.common_layers
.context_parallel`) each rank of a DCP group stores only the KV positions it
owns, and every step attends all of the group's query heads over each rank's
share of the keys, then merges the partial outputs with an all-to-all of
``T x H x D`` per layer. In a long prefill that merge and the per-head work
for every token dominate. PCP splits the step's *queries* instead, inside
the attention core only:

1. Each rank of a DCP group keeps a contiguous run of the batch's flat rows
   (the ceiling-first split a reduce-scatter makes: :func:`local_row_count`)
   and runs the query, latent and indexer projections over those rows only.
2. The group all-gathers the chunk's raw latent rows and indexer keys, and
   every rank stores the positions it owns into its DCP cache, so the
   persistent cache keeps DCP's layout exactly.
3. Every rank also builds a whole-context *scratch* cache outside any DCP
   group: the request's cached prefix, gathered from the ranks that own it
   (``kv_cache.pcp_pack_rows`` / ``kv_cache.pcp_unpack_rows``), and this
   chunk. The existing non-DCP sparse indexer and MLA prefill ops attend the
   rank's queries against it, each request's cache length shifted by the
   rows of it that precede the rank's (:meth:`PrefillContextParallel
   .query_view`), so RoPE and the causal mask see global positions. The
   indexer's top-k is local: there is no score merge.
4. One all-to-all along the group moves the attention output from
   token-sharded to head-sharded (:meth:`PrefillContextParallel
   .to_head_sharded`), the layout the DCP merge leaves, so ``o_proj`` and
   everything after it are unchanged.

The split math and the query view's position shift come from the prefill
context parallelism of disaggregated prefill nodes (CENG-1398).

A contiguous split rather than DCP's round-robin keeps each request's rows
on a rank one contiguous run, which the ragged prefill kernels address as
``cache_length + token``; the indexer and MLA ops then take a plain cache
length shift, not a position map. The causal imbalance it leaves is bounded:
sparse attention reads at most ``index_topk`` keys per query, so only the
indexer's scoring grows with position, and its worst rank still scores
fewer keys than every rank does without the split.

The prefill graph chooses this path per step with :func:`prefill_predicate`:
decode and verify steps, a few rows per request, stay on the DCP path.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any, overload

from max import functional as F
from max import tree
from max._core.dialects import kgen, mo
from max.driver import CPU, Device
from max.dtype import DType
from max.graph import (
    BufferType,
    BufferValue,
    DeviceRef,
    Dim,
    DimLike,
    Graph,
    TensorType,
    TensorValue,
    ops,
)
from max.nn.common_layers.context_parallel import (
    dcp_ranks,
    dcp_view,
    to_dcp_mesh,
)
from max.nn.common_layers.kv_cache import PagedCacheValues
from max.nn.common_layers.mesh_axis import DCP, TP, dcp_axis
from max.nn.kernels import (
    kv_cache_pack_owned_rows,
    kv_cache_row_bytes,
    kv_cache_unpack_prefix_rows,
)
from max.nn.kv_cache.cache_params import PACKED_PAGE_STRIDE
from max.tensor import Tensor
from max.tensor.sharding import (
    DeviceMapping,
    DeviceMesh,
    Placement,
    Sharded,
    Unknown,
)
from max.tensor.sharding.action import PerShard

PCP_MIN_ROWS_PER_REQUEST = 8
"""A step takes the PCP path when its rows average more than this per
request. Decode steps (one row) and the verify steps of every speculative
width a DCP capture serves (at most eight rows) stay on the DCP path. Mach
mirrors it (``PCP_MIN_ROWS_PER_REQUEST`` in ``kv_sizing.rs``, asserted
against ``DCP_MAX_CAPTURED_VERIFY_ROW`` in ``config.rs``) so its captured
verify width stays at or below it and a captured step never gives the PCP
path its rows."""

PCP_MIN_STEP_TOKENS = 7168
"""A step takes the PCP path only with at least this many rows. PCP pays a
mostly fixed cost per layer (the prefix fill and the row exchange): on
8xB200 a PCP8 prefill step takes about 305 ms from 500 to 8000 new tokens,
with or without a 32K prefix, while a DCP8 step grows from 87 ms (500)
through 195 ms (4000) to 347 ms (8000). They cross near 7000 tokens, so
shorter steps, chunked-prefill tails included, stay on the DCP path. These
times were measured on the earlier build that branched (``ops.cond``) per
layer; the predicated build inherits the crossover without re-measuring
it."""

PCP_MAX_BATCH_CONTEXT = 262144
"""A step takes the PCP path only while every replica's page-aligned summed
context is at most this many tokens. Each layer's whole-context scratch
holds about 1 KB per context token per device, so this bounds it near
270 MB, and keeps the sparse prefill on its convert-once arm (whose NVFP4
conversion takes at most this context)."""

__all__ = [
    "PCP_MAX_BATCH_CONTEXT",
    "PCP_MIN_ROWS_PER_REQUEST",
    "PCP_MIN_STEP_TOKENS",
    "PrefillContextParallel",
    "PrefillContextParallelStep",
    "local_row_count",
    "plan_prefill_context_parallel",
    "prefill_predicate",
    "rank_layout",
    "restore_order_index",
    "row_ranges",
    "rows_carrier",
    "split_rows",
]


@overload
def local_row_count(total: int, rank: int, size: int) -> int: ...


@overload
def local_row_count(total: Dim, rank: int, size: int) -> Dim: ...


def local_row_count(total: Dim | int, rank: int, size: int) -> Dim | int:
    """Returns rank ``rank``'s rows of ``total`` split ceiling-first.

    The first ``total % size`` ranks take one row more, as
    :func:`~max.tensor.sharding.placements.even_shard_sizes` does.
    """
    return (total + (size - 1 - rank)) // size


def rank_layout(offsets: Tensor, rank: int, size: int) -> tuple[Tensor, Tensor]:
    """Returns one rank's row offsets and per-request query starts.

    ``offsets`` are the batch's ``[B + 1]`` uint32 row offsets. The rank keeps
    the flat rows ``[start, end)`` of the ceiling-first split
    (:func:`local_row_count`). Its ``[B + 1]`` row offsets index its own rows,
    and its ``[B]`` query starts count each request's rows before them, so
    request ``b``'s first local row sits at position ``cache_length[b] +
    query_starts[b]``. A request with no rows on the rank is empty there.
    """
    device = offsets.device

    def const(value: int) -> Tensor:
        return F.constant(value, DType.uint32, device=device)

    total = F.max(offsets, axis=0)
    base = total // const(size)
    rem = total % const(size)
    start = base * const(rank) + F.min(rem, const(rank))
    end = start + base + F.cast(F.greater(rem, const(rank)), DType.uint32)
    row_offsets = F.min(F.max(offsets, start), end) - start
    request_starts = offsets[:-1]
    query_starts = (
        F.min(F.max(start, request_starts), offsets[1:]) - request_starts
    )
    return row_offsets, query_starts


def restore_order_index(total: Dim, size: int, device: Device) -> Tensor:
    """Returns where each flat row lands in the padded all-to-all output.

    Every rank pads its ``local_row_count`` rows to the first rank's count,
    ``T0 = ceil(T / size)``, before the all-to-all, so the receiving rank
    holds rank ``r``'s rows at ``[r * T0, r * T0 + rows_r)``. Row ``t`` of the
    batch, local row ``t - start_r`` of its rank ``r``, is at ``t`` plus the
    pad rows before it: one per rank past the first ``T % size``, which take
    ``T0`` rows each.
    """
    # All on the device: a host copy here would sync each layer's prefill
    # stream, and the host's launches would then trail the GPU.
    count = F.shape_to_tensor([total]).reshape([])
    t = F.range(0, count, 1, out_dim=total, dtype=DType.int32, device=device)
    i32 = lambda v: F.constant(v, DType.int32, device=device)
    n = F.max(t, axis=0) + i32(1)
    base = n // i32(size)
    rem = n % i32(size)
    first = base + F.cast(F.greater(rem, i32(0)), DType.int32)
    pad_d = first - base
    divisor = F.max(base, i32(1))
    boundary = rem * first
    past = F.max(t - boundary, i32(0))
    return t + pad_d * (past // divisor)


def prefill_predicate(
    tokens: Tensor,
    input_row_offsets: Tensor,
    batch_context_length: Tensor,
) -> Tensor:
    """Returns the host bool that routes a step to the PCP path.

    A step qualifies when its rows average more than
    :data:`PCP_MIN_ROWS_PER_REQUEST` per request (a prefill) and number at
    least :data:`PCP_MIN_STEP_TOKENS` (one long enough to repay PCP), and
    its summed context is at most :data:`PCP_MAX_BATCH_CONTEXT`. Both paths
    are exact for any batch; the predicate only picks the faster one. PCP
    runs at one data-parallel replica only (:func:`_assert_one_replica`).

    Args:
        tokens: The batch's ``[T]`` tokens, for ``T``.
        input_row_offsets: The batch's ``[B + 1]`` row offsets, for ``B``.
        batch_context_length: The ``[1]`` host bound on the batch's summed
            context, page-aligned per request, one shard per device.
    """
    host = CPU()
    i64 = lambda v: F.constant(v, DType.int64, device=host)
    # Every device's shard holds the same bound; the largest is any one.
    bounds = [
        F.reshape(F.cast(shard, DType.int64), [1])
        for shard in batch_context_length.local_shards
    ]
    fits = F.greater_equal(
        i64(PCP_MAX_BATCH_CONTEXT), F.max(F.concat(bounds), axis=0)
    ).reshape([])
    shape = F.shape_to_tensor([tokens.shape[0], input_row_offsets.shape[0]])
    total, requests = shape[0], shape[1] - i64(1)
    return F.logical_and(
        F.logical_and(
            F.greater(total, requests * i64(PCP_MIN_ROWS_PER_REQUEST)),
            F.greater_equal(total, i64(PCP_MIN_STEP_TOKENS)),
        ),
        fits,
    ).reshape([])


def _host_scalar(t: Tensor) -> Tensor:
    """Returns a ``[1]`` host tensor as an int64 scalar."""
    return F.cast(F.reshape(t, []), DType.int64)


def _host_row_range(total: Dim, rank: int, size: int) -> tuple[Tensor, Tensor]:
    """Returns rank ``rank``'s ``[start, stop)`` of ``total`` rows on the host.

    The ceiling-first split of :func:`local_row_count`, as int64 scalars.
    """
    host = CPU()
    count = F.shape_to_tensor([total]).reshape([])
    n = F.constant(size, DType.int64, device=host)
    r = F.constant(rank, DType.int64, device=host)
    base = count // n
    rem = count % n
    start = base * r + F.min(rem, r)
    stop = start + base + F.cast(F.greater(rem, r), DType.int64)
    return start, stop


_INT64_MAX = 2**63 - 1


@tree.dataclass
class RowBounds:
    """A rank-2 slice's host bounds, built once per step for every layer.

    ``[start, 0]``, ``[stop, INT64_MAX]`` and ``[1, 1]``: rows ``[start,
    stop)`` and every column. :func:`_slice_rows` takes them as they are, so
    a layer's slice builds no bounds on the host.
    """

    start: Tensor
    stop: Tensor
    step: Tensor


def _row_bounds(start: Tensor, stop: Tensor) -> RowBounds:
    """Returns the :class:`RowBounds` of rows ``[start, stop)``.

    ``start`` and ``stop`` are int64 host scalars.
    """
    host = CPU()
    i64 = lambda v: F.constant([v], DType.int64, device=host)
    return RowBounds(
        start=F.concat([F.reshape(start, [1]), i64(0)]),
        stop=F.concat([F.reshape(stop, [1]), i64(_INT64_MAX)]),
        step=F.constant([1, 1], DType.int64, device=host),
    )


# TODO(AGENT-800): use a public slice op once one takes prebuilt bounds.
def _slice_rows(v: Tensor, bounds: RowBounds, rows: DimLike) -> Tensor:
    """Returns the rows of rank-2 ``v`` within ``bounds``, ``rows`` of them.

    A view, as :func:`max.functional.slice_tensor`'s, whose bounds the
    caller built once: that op builds them from scalars at every call.
    Graph ops take no prebuilt bounds, so this emits the slice op itself.
    """
    value = TensorValue(v)
    if value.rank != 2:
        raise ValueError(f"row bounds slice rank-2 tensors, not {value.type}")
    result = TensorType(value.dtype, [rows, value.shape[1]], value.device)
    sliced = Graph.current._add_op_generated(
        mo.SliceOp,
        result=result,
        input=value,
        start=TensorValue(bounds.start),
        stop=TensorValue(bounds.stop),
        step=TensorValue(bounds.step),
        output_param_decls=kgen.ParamDeclArrayAttr([]),
    )[0]
    return Tensor.from_graph_value(sliced.tensor)


@tree.dataclass
class PrefillContextParallel:
    """One prefill step's split over each DCP group, shared by every layer.

    Built once per step by :func:`plan_prefill_context_parallel`. Tensors are
    on the model mesh; a value that differs between a group's ranks is
    ``Unknown`` on the ``tp`` axis.
    """

    dcp: int
    """The DCP group size."""

    interleave: int
    """Consecutive positions a DCP rank owns in turn."""

    namespace: str
    """Prefix of the symbolic dims the plan declares, one plan's own."""

    batch_row_offsets: Tensor
    """Each replica's whole ``[B + 1]`` row offsets."""

    row_offsets: Tensor
    """Each rank's ``[B + 1]`` offsets into its own rows."""

    query_starts: Tensor
    """Each rank's ``[B]`` count of each request's rows before its own."""

    row_start: Tensor
    """Each rank's first flat row, an int64 host scalar
    (:func:`row_ranges`)."""

    row_stop: Tensor
    """Each rank's flat row past its last, an int64 host scalar."""

    rows: tuple[Tensor, ...]
    """Each DCP rank's ``[rows]`` uint8 host carrier of its row count: the
    one declaration of its ``{namespace}_rows_{rank}`` symbol, which every
    layer's :meth:`local_rows` and :meth:`own_rows` slice rebinds to. One
    per rank, not per device: a group's ranks hold the same counts on every
    group, and two declarations of one value would be unified to one
    symbol under a subgraph signature that names both. A symbol a layer
    declared inside its PCP branch would be bound there at run time by a
    host shape function, which keeps every buffer it sizes out of the
    memory plan and leaves each of the branch's kernels its own primitive
    (a host launch-rate floor per layer)."""

    restore: Tensor
    """``[T]`` int32: each flat row's index in the padded all-to-all output
    (:func:`restore_order_index`)."""

    prefix_offsets: Tensor
    """``[B + 1]`` uint32: each request's first packed prefix row on every
    rank (:func:`max.nn.kernels.kv_cache_pack_owned_rows`)."""

    prefix_rows: Tensor
    """``[N]`` int32 whose length is the packed prefix rows per rank: at
    least ``prefix_offsets[B]`` and at least 1 on a PCP step, 0 on any
    other step, where the prefix pack, gather and unpack launch nothing."""

    mla_scratch: PagedCacheValues
    """The whole-context latent cache of this step's requests, one layer."""

    indexer_scratch: PagedCacheValues
    """The whole-context indexer key cache, one layer."""

    mla_view_fields: dict[str, Any] | None = None
    """The fields :meth:`query_view` changes in :attr:`mla_scratch`, built
    once with the plan. The view is rebuilt from them where a layer reads it
    (:attr:`mla_view`), so a layer takes the scratch buffers once: a subgraph
    that took them as two parameters would assume they do not alias."""

    indexer_view_fields: dict[str, Any] | None = None
    """The same as :attr:`mla_view_fields`, for :attr:`indexer_scratch`."""

    zero_prefill_offsets: Tensor | None = None
    """``[1, B + 1]`` uint32 zeros on each device, built once with the plan:
    the dense prefill plan's row and cache offsets, which the sparse prefill
    op types but never reads. Built per layer, they would launch two fills
    on every device in every layer of every step."""

    local_bounds: tuple[RowBounds, ...] = ()
    """Each DCP rank's :attr:`row_start` to :attr:`row_stop` as
    :class:`RowBounds`, built once with the plan, so a layer's
    :meth:`local_rows` of a rank-2 tensor is one view."""

    active_rows: Tensor | None = None
    """A ``[1]`` uint8 host carrier on a PCP step and ``[0]`` otherwise: the
    rows of :meth:`fill_prefix`'s spare row, so an off step gathers none."""

    @property
    def mla_view(self) -> PagedCacheValues:
        """:attr:`mla_scratch` as each rank's queries see it."""
        assert self.mla_view_fields is not None
        return dataclasses.replace(self.mla_scratch, **self.mla_view_fields)

    @property
    def indexer_view(self) -> PagedCacheValues:
        """:attr:`indexer_scratch` as each rank's queries see it."""
        assert self.indexer_view_fields is not None
        return dataclasses.replace(
            self.indexer_scratch, **self.indexer_view_fields
        )

    @property
    def mesh(self) -> DeviceMesh:
        """The model mesh."""
        return self.row_offsets.mesh

    @property
    def view(self) -> DeviceMesh:
        """The model mesh's DCP view (:func:`dcp_view`)."""
        return dcp_view(self.mesh, self.dcp)

    def _rows_mapping(self, mesh: DeviceMesh, like: Tensor) -> DeviceMapping:
        """``like``'s mapping on ``mesh`` with rows sharded along DCP.

        On the model mesh that is the ``tp`` axis: exact when a DCP group
        spans it, and when it does not the local rows are only ever read by
        row-wise ops with weights replicated over ``tp``, then by this
        class's collectives over each group, so no rule moves them.
        """
        axis = dcp_axis(mesh)
        placements: list[Placement] = list(like.placements)
        if len(placements) != mesh.ndim:
            # ``like`` on the model mesh, ``mesh`` its view: the view's extra
            # axis takes ``like``'s ``tp`` placement first.
            placements = list(to_dcp_mesh(like, mesh).placements)
        placements[axis] = Sharded(0)
        return DeviceMapping(mesh, tuple(placements))

    def local_rows(self, x: Tensor) -> Tensor:
        """Returns each rank's run of the batch's rows of ``x``.

        ``x`` is ``[T, ...]``, the same on every rank of a group.
        """
        n = x.mesh.num_devices
        if x.rank == 2 and self.local_bounds:
            ranks = self._ranks(x.mesh)
            return F.call_on_mesh(
                lambda v, lo, hi, step, rows: _slice_rows(
                    v, RowBounds(lo, hi, step), rows.shape[0]
                ),
                x.mesh,
                out_specs=self._rows_mapping(x.mesh, x),
            )(
                x,
                PerShard(self.local_bounds[r].start for r in ranks),
                PerShard(self.local_bounds[r].stop for r in ranks),
                PerShard(self.local_bounds[r].step for r in ranks),
                self._rank_rows(x.mesh),
            )

        def slice_rows(
            x: Tensor, start: Tensor, stop: Tensor, rows: Tensor
        ) -> Tensor:
            # The plan's symbol (:attr:`rows`), not ``local_row_count``'s
            # expression: the rows' top-k crosses the layer subgraphs'
            # boundaries, whose signatures take only symbols. Every layer
            # slices the same rows.
            return F.slice_tensor(
                x,
                [(slice(start, stop), rows.shape[0])]
                + [slice(None)] * (x.rank - 1),
            )

        return F.call_on_mesh(
            slice_rows, x.mesh, out_specs=self._rows_mapping(x.mesh, x)
        )(
            x,
            _per_device_host(self.row_start, n),
            _per_device_host(self.row_stop, n),
            self._rank_rows(x.mesh),
        )

    def own_rows(self, t: Tensor) -> Tensor:
        """Returns the first ``local_rows`` rows of each device's ``t``.

        ``t`` is a ``[T, ...]`` per-device tensor whose leading rows are the
        rank's own.
        """
        n = t.mesh.num_devices

        def head(
            t: Tensor, start: Tensor, stop: Tensor, rows: Tensor
        ) -> Tensor:
            zero = F.constant(0, DType.int64, device=CPU())
            return F.slice_tensor(
                t,
                [(slice(zero, stop - start), rows.shape[0])]
                + [slice(None)] * (t.rank - 1),
            )

        return F.call_on_mesh(head, t.mesh, out_specs=t.mapping)(
            t,
            _per_device_host(self.row_start, n),
            _per_device_host(self.row_stop, n),
            self._rank_rows(t.mesh),
        )

    def _rank_rows(self, mesh: DeviceMesh) -> PerShard[Tensor]:
        """Each device's rank's carrier (:attr:`rows`), for ``call_on_mesh``."""
        return PerShard(self.rows[rank] for rank in self._ranks(mesh))

    def _ranks(self, mesh: DeviceMesh) -> PerShard[int]:
        return dcp_ranks(mesh, self.dcp)

    def gather_rows(self, x: Tensor, like: Tensor) -> Tensor:
        """All-gathers every rank's rows of ``x`` within each DCP group.

        Returns the batch's ``[T, ...]`` rows in batch order on every rank,
        placed like ``like``, a tensor of the whole batch's rows.
        """
        mesh = x.mesh
        view = self.view
        rows = to_dcp_mesh(x, view)
        axis = dcp_axis(view)
        placements: list[Placement] = [Unknown()] * view.ndim
        placements[axis] = Sharded(0)
        gathered = F.allgather(
            rows.rebind_mapping(DeviceMapping(view, tuple(placements))),
            tensor_axis=0,
            mesh_axis=axis,
        )
        gathered = to_dcp_mesh(
            gathered.rebind_mapping(
                DeviceMapping(view, (Unknown(),) * view.ndim)
            ),
            mesh,
        )
        total = self.restore.shape[0]
        return F.call_on_mesh(
            lambda g, n: F.rebind(g, [n, *g.shape[1:]]),
            mesh,
            out_specs=like.mapping,
        )(gathered, total)

    def to_head_sharded(self, out: Tensor, like: Tensor) -> Tensor:
        """Moves each rank's ``[rows, H, D]`` output to its heads of all rows.

        ``H`` is the group's heads. Returns ``[T, H / dcp, D]``: every row of
        the batch in order, and rank ``j`` of each group its run ``j`` of the
        group's heads, the layout
        :func:`~max.nn.common_layers.context_parallel.combine_partial_attention`
        returns: on a mesh's ``tp`` axis, each rank's tensor-parallel heads.
        Other mesh axes take ``like``'s placements, a tensor of the whole
        batch's rows.
        """
        mesh = out.mesh
        view = self.view
        dcp = self.dcp

        def pad(o: Tensor, rank: int, total: Dim) -> Tensor:
            first = local_row_count(total, 0, dcp)
            missing = first - local_row_count(total, rank, dcp)
            zeros = F.full(
                [missing, *o.shape[1:]], 0, dtype=o.dtype, device=o.device
            )
            return F.rebind(F.concat([o, zeros], axis=0), [first, *o.shape[1:]])

        padded = F.call_on_mesh(
            pad, mesh, out_specs=DeviceMapping(mesh, (Unknown(),) * mesh.ndim)
        )(out, self._ranks(mesh), self.restore.shape[0])
        axis = dcp_axis(view)
        received = F.all_to_all(
            to_dcp_mesh(padded, view),
            split_axis=1,
            concat_axis=0,
            mesh_axis=axis,
        )
        # Group ``g`` attended heads ``[g * H, (g + 1) * H)``, and rank ``j``
        # now holds run ``j`` of them: its slice along the whole ``tp`` axis.
        heads = DeviceMapping(
            view,
            tuple(
                Sharded(1) if name in (TP, DCP) else p
                for name, p in zip(
                    view.axis_names,
                    to_dcp_mesh(like, view).placements,
                    strict=True,
                )
            ),
        )
        restored = F.call_on_mesh(
            lambda r, idx: F.gather(r, idx, axis=0),
            view,
            out_specs=heads,
        )(received, to_dcp_mesh(self.restore, view))
        return to_dcp_mesh(restored, mesh)

    def query_view(self, kv_collection: PagedCacheValues) -> PagedCacheValues:
        """Returns a cache as each rank's queries see it."""
        return dataclasses.replace(
            kv_collection, **self.query_view_fields(kv_collection)
        )

    def query_view_fields(
        self, kv_collection: PagedCacheValues
    ) -> dict[str, Any]:
        """Returns the fields of a cache that its query view changes.

        The chunk is stored at its global positions; shifting each request's
        cache length by :attr:`query_starts` places a rank's first query of
        it at its global position. The longest context grows by up to a
        prompt, so the bound does too.
        """

        def shift(lengths: Tensor, starts: Tensor) -> Tensor:
            # The cache's batch dim and the row offsets' are separate symbols
            # that agree at runtime.
            return lengths + F.rebind(starts, lengths.shape)

        mesh = kv_collection.cache_lengths.mesh
        cache_lengths = F.call_on_mesh(
            shift, mesh, out_specs=DeviceMapping(mesh, (Unknown(),) * mesh.ndim)
        )(kv_collection.cache_lengths, self.query_starts)
        return {
            "cache_lengths": cache_lengths,
            "max_cache_length": kv_collection.max_cache_length
            + kv_collection.max_prompt_length,
            **self._dispatch_fields(kv_collection),
        }

    def _dispatch_fields(
        self, kv_collection: PagedCacheValues
    ) -> dict[str, Any]:
        """The decode dispatch records of each rank's own rows.

        The DCP cache's records describe the whole batch's rows over each
        rank's share of the context; an MLA op that falls back to its decode
        route over the scratch reads them. Each rank's record keeps the
        batch size and the DCP record's partition count (a valid split of
        any longer context) and takes the rank's own longest query. The
        mixed-step split, made for the whole batch's rows, is turned off.
        The sparse dispatch never adjusts the batch size it records.
        """
        names = [
            name
            for name in (
                "mixed_step_split",
                "mla_num_partitions",
                "attention_dispatch_metadata",
            )
            if getattr(kv_collection, name) is not None
        ]
        if not names:
            return {}
        records = {name: getattr(kv_collection, name) for name in names}
        unknown = DeviceMapping(self.mesh, (Unknown(),) * self.mesh.ndim)
        specs = [
            unknown
            if name == "attention_dispatch_metadata"
            else records[name].mapping
            for name in names
        ]

        def device_fields(
            records: dict[str, Tensor], offsets: Tensor
        ) -> list[Tensor]:
            out = []
            for name in names:
                v = records[name]
                if name == "mixed_step_split":
                    out.append(
                        F.constant(
                            [0] * int(v.shape[0]), DType.uint32, device=v.device
                        )
                    )
                elif name == "mla_num_partitions":
                    zero = F.constant([0], DType.int64, device=v.device)
                    out.append(F.concat([v[:1], zero]))
                else:
                    lengths = F.cast(offsets[1:] - offsets[:-1], DType.int64)
                    longest = F.max(lengths, axis=0)
                    zeros = F.constant(
                        [0] * (int(v.shape[0]) - 3),
                        DType.int64,
                        device=v.device,
                    )
                    out.append(
                        F.rebind(
                            F.concat([v[:1], longest, v[2:3], zeros]), v.shape
                        )
                    )
            return out

        values = F.call_on_mesh(device_fields, self.mesh, out_specs=specs)(
            records, self.row_offsets
        )
        return dict(zip(names, values, strict=True))

    def fill_prefix(
        self,
        pairs: Sequence[tuple[PagedCacheValues, PagedCacheValues, Tensor]],
    ) -> None:
        """Copies every request's cached prefix into the scratch caches.

        Each ``(cache, scratch, layer_idx)`` names a DCP cache, the scratch
        cache it fills, and the cache's layer. Each rank packs the prefix
        rows it owns of every pair into one byte row per position slot, the
        group all-gathers them once, and each rank writes every position
        into the scratch caches. A no-op when no request has a prefix.
        """
        widths = [kv_cache_row_bytes(cache) for cache, _, _ in pairs]
        width = sum(widths)
        offsets = [sum(widths[:i]) for i in range(len(widths))]
        dcp, interleave = self.dcp, self.interleave
        mesh = self.mesh
        view = self.view
        axis = dcp_axis(view)
        caches = [cache for cache, _, _ in pairs]
        scratches = [scratch for _, scratch, _ in pairs]
        layers = [layer for _, _, layer in pairs]

        def gather_and_fill() -> None:
            def pack(
                prefix_rows: Tensor,
                prefix_offsets: Tensor,
                rank: int,
                caches: list[PagedCacheValues],
            ) -> Tensor:
                # A raw buffer: there is no uninitialized ``Tensor`` to write.
                packed = ops.buffer_create(
                    BufferType(
                        DType.uint8,
                        [prefix_rows.shape[0], width],
                        device=DeviceRef.from_device(prefix_offsets.device),
                    )
                )
                for cache, layer, offset in zip(
                    caches, layers, offsets, strict=True
                ):
                    kv_cache_pack_owned_rows(
                        packed,
                        prefix_offsets,
                        cache,
                        layer,
                        col_offset=offset,
                        dcp_size=dcp,
                        dcp_interleave=interleave,
                        dcp_rank=rank,
                    )
                # The all-gather takes an immutable tensor, and a bare load
                # of the written buffer folds back into it. One more zero
                # row makes the rows a new tensor; the unpack splits the
                # gathered rows evenly by rank, so it is skipped there.
                spare_rows = (
                    1 if self.active_rows is None else self.active_rows.shape[0]
                )
                spare = F.full(
                    [spare_rows, width],
                    0,
                    dtype=DType.uint8,
                    device=prefix_offsets.device,
                )
                loaded = Tensor.from_graph_value(ops.buffer_load(packed))
                return F.concat([loaded, spare], axis=0)

            unknown = DeviceMapping(mesh, (Unknown(),) * mesh.ndim)
            packed = F.call_on_mesh(pack, mesh, out_specs=unknown)(
                self.prefix_rows,
                self.prefix_offsets,
                self._ranks(mesh),
                caches,
            )
            placements: list[Placement] = [Unknown()] * view.ndim
            placements[axis] = Sharded(0)
            gathered = F.allgather(
                to_dcp_mesh(packed, view).rebind_mapping(
                    DeviceMapping(view, tuple(placements))
                ),
                tensor_axis=0,
                mesh_axis=axis,
            ).rebind_mapping(DeviceMapping(view, (Unknown(),) * view.ndim))
            gathered = to_dcp_mesh(gathered, mesh)
            scratch_layer = F.constant(0, DType.uint32, device=CPU())
            for scratch, offset in zip(scratches, offsets, strict=True):
                kv_cache_unpack_prefix_rows(
                    gathered,
                    self.prefix_offsets,
                    scratch,
                    scratch_layer,
                    col_offset=offset,
                    dcp_size=dcp,
                    dcp_interleave=interleave,
                )

        # Not under an ``ops.cond``: the plan sizes the carrier to no rows
        # on a step that does not run PCP, and to one row on a PCP step
        # without a prefix, where every offset is zero; either way the pack
        # and unpack write nothing.
        gather_and_fill()


@tree.dataclass
class PrefillContextParallelStep:
    """A step's choice of prefill context parallelism and its plan.

    The model predicates the step's rows (:meth:`predicated`), builds the
    plan once per step above its layers (:meth:`with_plan`), and passes this
    to them; each layer's attention runs the PCP path on the trailing
    :attr:`cut` rows and the DCP path on the rest. The plan is built on
    every step, sized to one scratch page and no prefix rows when the step
    does not run PCP, and takes no host copy, so a decode or verify step's
    capture records only its few small kernels.
    """

    dcp: int
    """The DCP group size."""

    interleave: int
    """Consecutive positions a DCP rank owns in turn."""

    page_size: int
    """Tokens per page of the scratch caches; the DCP caches'."""

    namespace: str
    """Prefix of the plan's symbolic dims (see
    :func:`plan_prefill_context_parallel`)."""

    active: Tensor
    """Host bool: whether this step runs PCP (:func:`prefill_predicate`)."""

    batch_context_length: Tensor
    """Each replica's host bound on its summed context, page-aligned per
    request (the MLA prefill plan's)."""

    split: PrefillContextParallel | None = None
    """The step's plan, when the model built it once (:meth:`with_plan`)."""

    cut: Tensor | None = None
    """Predicated steps (:meth:`predicated`): the int64 host count of the
    step's trailing rows the PCP path takes, all of them on a PCP step and
    none otherwise. The DCP path takes the leading rest, a slice from row 0
    and so a view, which keeps a decode step's DCP path copy-free. Both
    paths run every step, and the empty one launches nothing."""

    pcp_rows: Tensor | None = None
    """A ``[cut]`` uint8 host carrier, the ``{namespace}_pcp_rows`` symbol."""

    dcp_rows: Tensor | None = None
    """A ``[T - cut]`` uint8 host carrier, the ``{namespace}_dcp_rows``
    symbol."""

    pcp_bounds: RowBounds | None = None
    """The PCP rows' :class:`RowBounds`, ``[T - cut, T)``, built once with
    :attr:`cut` so a layer's :meth:`rows_split` of a rank-2 tensor builds
    nothing on the host."""

    dcp_bounds: RowBounds | None = None
    """The DCP rows' :class:`RowBounds`, ``[0, T - cut)``."""

    def predicated(self, tokens: Tensor) -> PrefillContextParallelStep:
        """Returns this step with the PCP and DCP paths' row split.

        ``tokens`` is any ``[T, ...]`` tensor of the step's rows. With one
        replica every device holds all of them, so the first device's count
        is every device's.
        """
        _assert_one_replica(tokens.mesh)
        rows = tokens.local_shards[0]
        total = F.shape_to_tensor([rows.shape[0]]).reshape([])
        cut = total * F.cast(self.active, DType.int64)

        def carrier(count: Tensor, name: str) -> Tensor:
            return _carrier(F.reshape(count, [1]), f"{self.namespace}_{name}")

        zero = F.constant(0, DType.int64, device=CPU())
        return dataclasses.replace(
            self,
            cut=cut,
            pcp_rows=carrier(cut, "pcp_rows"),
            dcp_rows=carrier(total - cut, "dcp_rows"),
            pcp_bounds=_row_bounds(total - cut, total),
            dcp_bounds=_row_bounds(zero, total - cut),
        )

    def rows_split(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Returns ``x``'s PCP rows (the trailing :attr:`cut`) and DCP rows.

        The DCP rows lead, so their slice starts at row 0. Every device
        slices at the one :attr:`cut`, which holds for one replica only.
        """
        _assert_one_replica(x.mesh)
        assert self.cut is not None
        assert self.pcp_rows is not None and self.dcp_rows is not None
        cut = TensorValue(self.cut)
        pcp_dim = TensorValue(self.pcp_rows).shape[0]
        dcp_dim = TensorValue(self.dcp_rows).shape[0]
        if x.rank == 2 and self.pcp_bounds is not None:
            pcp_b, dcp_b = self.pcp_bounds, self.dcp_bounds
            assert dcp_b is not None
            return F.call_on_mesh(
                lambda v: (
                    _slice_rows(v, pcp_b, pcp_dim),
                    _slice_rows(v, dcp_b, dcp_dim),
                ),
                x.mesh,
                out_specs=[x.mapping, x.mapping],
            )(x)

        def split(v: TensorValue) -> tuple[TensorValue, TensorValue]:
            v = TensorValue(v)
            zero = ops.constant(0, DType.int64, device=DeviceRef.CPU())
            total = ops.shape_to_tensor([v.shape[0]]).reshape([])
            rest = [slice(None)] * (v.rank - 1)
            return (
                ops.slice_tensor(
                    v, [(slice(total - cut, total), pcp_dim), *rest]
                ),
                ops.slice_tensor(
                    v, [(slice(zero, total - cut), dcp_dim), *rest]
                ),
            )

        first, second = F.call_on_mesh(
            split, x.mesh, out_specs=[x.mapping, x.mapping]
        )(x)
        return first, second

    def rows_join(self, pcp: Tensor, dcp: Tensor, like: Tensor) -> Tensor:
        """Joins :meth:`rows_split`'s two parts back to ``like``'s rows.

        ``pcp`` holds the PCP rows and ``dcp`` the DCP rows; the DCP rows go
        first.
        """
        if pcp.mapping != dcp.mapping:
            raise ValueError(
                f"the PCP and DCP paths place their rows differently:"
                f" {pcp.mapping} and {dcp.mapping}"
            )

        def join(p: Tensor, d: Tensor, x: Tensor) -> Tensor:
            return F.rebind(
                F.concat([d, p], axis=0), [x.shape[0], *p.shape[1:]]
            )

        return F.call_on_mesh(join, like.mesh, out_specs=pcp.mapping)(
            pcp, dcp, like
        )

    def with_plan(
        self,
        input_row_offsets: Tensor,
        tokens: Tensor,
        kv_collection: PagedCacheValues,
        indexer_kv_collection: PagedCacheValues,
    ) -> PrefillContextParallelStep:
        """Returns this step with its plan, built once for every layer."""
        return dataclasses.replace(
            self,
            split=self.plan(
                input_row_offsets, tokens, kv_collection, indexer_kv_collection
            ),
        )

    def plan(
        self,
        input_row_offsets: Tensor,
        tokens: Tensor,
        kv_collection: PagedCacheValues,
        indexer_kv_collection: PagedCacheValues,
    ) -> PrefillContextParallel:
        """Returns the step's plan: :attr:`split`, or a new one.

        Every layer reads the same plan; the scratch caches it allocates
        hold one layer at a time.
        """
        if self.split is not None:
            return self.split
        return plan_prefill_context_parallel(
            input_row_offsets,
            tokens,
            kv_collection,
            indexer_kv_collection,
            self.batch_context_length,
            self.dcp,
            self.interleave,
            self.page_size,
            self.active,
            namespace=self.namespace,
        )


def _per_device_host(t: Tensor, num_devices: int) -> PerShard[Tensor]:
    """Each device's value of a host tensor given once.

    The tensor holds one shard per replica or per device: device ``i`` of
    ``num_devices`` reads shard ``i // (num_devices // shards)``.
    """
    shards = t.local_shards
    if num_devices % len(shards):
        raise ValueError(
            f"{len(shards)} host values do not split {num_devices} devices"
        )
    group = num_devices // len(shards)
    return PerShard(shards[i // group] for i in range(num_devices))


def _scratch_cache(
    collection: PagedCacheValues, pages: Dim, lookup: Tensor
) -> PagedCacheValues:
    """Builds a one-layer whole-context cache shaped like ``collection``.

    Its pages hold a step's requests outside any DCP group: request ``b``'s
    position ``p`` at its ``p``-th row through ``lookup``. Cache lengths and
    host bounds are the DCP cache's, which are already global.
    """
    mesh = collection.cache_lengths.mesh
    scaled = collection.kv_scales is not None
    host = CPU()

    def device_scratch(cache: PagedCacheValues) -> list[Any]:
        # Raw buffers: there is no uninitialized ``Tensor`` to write, and
        # zeroing a whole-context cache would cost a memset per step.
        def like(blocks: Tensor) -> BufferValue:
            return ops.buffer_create(
                BufferType(
                    blocks.dtype,
                    [pages, blocks.shape[1], 1, *blocks.shape[3:]],
                    device=DeviceRef.from_device(blocks.device),
                )
            )

        stride = F.constant([PACKED_PAGE_STRIDE], DType.int64, device=host)
        out: list[Any] = [like(cache.kv_blocks), stride]
        if cache.kv_scales is not None:
            out += [like(cache.kv_scales), stride]
        return out

    # Each device's pages and table columns are its own symbols, so those
    # fields are ``Unknown``: replicated, a subgraph call would type every
    # device's buffers like device 0's. The page strides are host values.
    unknown = DeviceMapping(mesh, (Unknown(),) * mesh.ndim)
    strides = collection.cache_lengths.mapping
    made = F.call_on_mesh(
        device_scratch,
        mesh,
        out_specs=[unknown, strides] * (2 if scaled else 1),
    )(collection)
    kv_scales = scales_page_stride = scales_lookup_table = None
    if scaled:
        kv_scales, scales_page_stride = made[2], made[3]
        scales_lookup_table = lookup
    return dataclasses.replace(
        collection,
        kv_blocks=made[0],
        page_stride=made[1],
        lookup_table=lookup,
        kv_scales=kv_scales,
        scales_page_stride=scales_page_stride,
        scales_lookup_table=scales_lookup_table,
    )


# TODO(AGENT-800): use F once it can declare a named dim from a host count.
def _carrier(count: Tensor, name: str) -> Tensor:
    """Returns a ``[count]`` uint8 host tensor declaring symbol ``name``.

    It is the one declaration of ``name``, whose value is the ``[1]`` int64
    host ``count``. Graph ops take no named output dim, so it is a raw op,
    as :func:`_slice_rows` is.
    """
    return Tensor.from_graph_value(
        ops.broadcast_to(
            ops.constant(0, DType.uint8, device=DeviceRef.CPU()),
            TensorValue(count),
            out_dims=[name],
        )
    )


def _assert_one_replica(mesh: DeviceMesh) -> None:
    """Asserts that ``mesh`` holds one data-parallel replica.

    The row split's symbols (the PCP and DCP row counts, each rank's rows)
    come from the first device, so they would hold only the first replica's
    rows. The config refuses PCP above one replica
    (``DeepseekV3_2Config``); per-replica splits are a follow-up.
    """
    replicas = 1
    for size, name in zip(mesh.mesh_shape, mesh.axis_names, strict=True):
        if name == "dp":
            replicas *= size
    assert replicas == 1, f"PCP takes one replica, not {replicas} ({mesh})"


def _host_mesh(mesh: DeviceMesh) -> DeviceMesh:
    """A host mirror of ``mesh``: its shape and axis names on the CPU."""
    return DeviceMesh(
        tuple(CPU() for _ in range(mesh.num_devices)),
        mesh.mesh_shape,
        mesh.axis_names,
    )


def row_ranges(tokens: Tensor, dcp: int) -> tuple[Tensor, Tensor]:
    """Returns each rank's ``[start, stop)`` of the batch's flat rows.

    ``tokens`` is any ``[T, ...]`` tensor of a replica's rows. The ranges
    are int64 host scalars, ``Unknown`` on a host mirror of its mesh: the
    ceiling-first split of :func:`local_row_count`.
    """
    mesh = tokens.mesh

    def device_range(rows: Tensor, rank: int) -> tuple[Tensor, Tensor]:
        return _host_row_range(rows.shape[0], rank, dcp)

    host = DeviceMapping(_host_mesh(mesh), (Unknown(),) * mesh.ndim)
    start, stop = F.call_on_mesh(device_range, mesh, out_specs=[host, host])(
        tokens, dcp_ranks(mesh, dcp)
    )
    return start, stop


def split_rows(
    input_row_offsets: Tensor, tokens: Tensor, dcp: int
) -> tuple[Tensor, Tensor, Tensor]:
    """Returns each rank's row offsets, query starts and the restore index.

    ``input_row_offsets`` are each replica's ``[B + 1]`` uint32 row offsets,
    the same on every rank of a replica, and ``tokens`` any ``[T, ...]``
    tensor of the replica's rows. Each result differs between the ranks of a
    DCP group, so it is ``Unknown`` on every axis: :func:`rank_layout`'s
    ``[B + 1]`` row offsets and ``[B]`` query starts, and
    :func:`restore_order_index`'s ``[T]`` index.
    """
    mesh = input_row_offsets.mesh

    def device_split(
        offsets: Tensor, rows: Tensor, rank: int
    ) -> tuple[Tensor, Tensor, Tensor]:
        row_offsets, query_starts = rank_layout(offsets, rank, dcp)
        restore = restore_order_index(rows.shape[0], dcp, offsets.device)
        return row_offsets, query_starts, restore

    unknown = DeviceMapping(mesh, (Unknown(),) * mesh.ndim)
    row_offsets, query_starts, restore = F.call_on_mesh(
        device_split, mesh, out_specs=[unknown, unknown, unknown]
    )(input_row_offsets, tokens, dcp_ranks(mesh, dcp))
    return row_offsets, query_starts, restore


def rows_carrier(
    row_start: Tensor,
    row_stop: Tensor,
    mesh: DeviceMesh,
    dcp: int,
    namespace: str,
) -> tuple[Tensor, ...]:
    """Returns each DCP rank's ``[rows]`` host carrier of its row count.

    Declares the ``{namespace}_rows_{rank}`` symbols, once per plan, for
    :attr:`PrefillContextParallel.rows`: a uint8 host tensor of ``row_stop -
    row_start`` bytes. Device ``rank`` is rank ``rank`` of the first DCP
    group (:func:`dcp_ranks`), and with one replica every group splits the
    same rows, so its count is every group's.
    """
    _assert_one_replica(mesh)
    starts = _per_device_host(row_start, mesh.num_devices)
    stops = _per_device_host(row_stop, mesh.num_devices)
    return tuple(
        _carrier(
            F.reshape(stops[rank] - starts[rank], [1]),
            f"{namespace}_rows_{rank}",
        )
        for rank in range(dcp)
    )


def plan_prefill_context_parallel(
    input_row_offsets: Tensor,
    tokens: Tensor,
    kv_collection: PagedCacheValues,
    indexer_kv_collection: PagedCacheValues,
    batch_context_length: Tensor,
    dcp: int,
    interleave: int,
    page_size: int,
    active: Tensor,
    namespace: str = "pcp",
) -> PrefillContextParallel:
    """Plans one prefill step's split and builds its scratch caches.

    Args:
        input_row_offsets: Each replica's ``[B + 1]`` uint32 row offsets,
            the same on every rank of a replica.
        tokens: Any ``[T, ...]`` tensor of the replica's rows, for ``T``.
        kv_collection: The DCP latent cache, global lengths.
        indexer_kv_collection: The DCP indexer key cache.
        batch_context_length: Each replica's host bound on its summed
            context, page-aligned per request (the MLA prefill plan's).
        dcp: The DCP group size.
        interleave: Consecutive positions a DCP rank owns in turn.
        page_size: Tokens per page of the scratch caches; the DCP caches'.
        active: The step's host bool (:func:`prefill_predicate`). The plan
            is built on every step (:meth:`PrefillContextParallelStep.
            with_plan`); where it is false, the plan sizes its scratch at
            one page and its prefix at no rows.
        namespace: Prefix of the symbolic dims this plan declares, so two
            plans in one graph (the target's and a draft's) don't collide.
    """
    mesh = input_row_offsets.mesh
    span = dcp * interleave

    def device_plan(
        offsets: Tensor,
        cache_lengths: Tensor,
        lookup: Tensor,
        bound: Tensor,
        max_cache: Tensor,
        on_host: Tensor,
        index: int,
    ) -> tuple[Tensor, ...]:
        device = offsets.device
        host = CPU()
        u32 = lambda v: F.constant(v, DType.uint32, device=device)
        i64 = lambda v: F.constant(v, DType.int64, device=host)

        # The scratch: request ``b``'s ``ceil(context / page)`` pages, in
        # order. ``bound`` sums each request's context rounded up to a page
        # at least this size, so it bounds the pages.
        lengths = offsets[1:] - offsets[:-1]
        cached = F.rebind(cache_lengths, lengths.shape)
        context = cached + lengths
        pages = (context + u32(page_size - 1)) // u32(page_size)
        first_page = F.cumsum(pages, exclusive=True)
        width = lookup.shape[1] * dcp
        width_t = F.shape_to_tensor([width]).reshape([])
        column = F.range(
            0, width_t, 1, out_dim=width, dtype=DType.uint32, device=device
        )
        last = F.max(pages, u32(1)) - u32(1)
        table = F.unsqueeze(first_page, 1) + F.min(
            F.unsqueeze(column, 0), F.unsqueeze(last, 1)
        )
        # Symbols, not expressions, for the layer subgraphs' signatures.
        table = F.rebind(
            table,
            [cache_lengths.shape[0], f"{namespace}_scratch_cols_{index}"],
        )
        on = F.cast(on_host, DType.int64)
        n_pages = i64(1) + on * (_host_scalar(bound) // i64(page_size) + i64(1))

        # The prefix each rank packs: ``max_local_len`` rows per request.
        local = (cached + u32(span - 1)) // u32(span) * u32(interleave)
        prefix_offsets = F.concat(
            [F.constant([0], DType.uint32, device=device), F.cumsum(local)]
        )
        requests = F.shape_to_tensor([lengths.shape[0]]).reshape([])
        longest = (
            (_host_scalar(max_cache) + i64(span - 1)) // i64(span)
        ) * i64(interleave)
        per_rank = F.min(
            _host_scalar(bound) // i64(dcp) + requests * i64(interleave),
            requests * longest,
        )
        # Off, no rows: the prefix pack, gather and unpack then launch nothing.
        n_rows = on * F.max(per_rank, i64(1))
        # A device range takes its length from the host scalar, so the
        # carrier needs no host copy.
        prefix_rows = F.range(
            i64(0),
            n_rows,
            i64(1),
            out_dim=f"{namespace}_prefix_rows_{index}",
            dtype=DType.int32,
            device=device,
        )
        return table, prefix_offsets, prefix_rows, F.reshape(n_pages, [1])

    unknown = DeviceMapping(mesh, (Unknown(),) * mesh.ndim)
    host_mesh = _host_mesh(mesh)
    row_offsets, query_starts, restore = split_rows(
        input_row_offsets, tokens, dcp
    )
    row_start, row_stop = row_ranges(tokens, dcp)
    rows = rows_carrier(row_start, row_stop, mesh, dcp, namespace)
    starts = _per_device_host(row_start, mesh.num_devices)
    stops = _per_device_host(row_stop, mesh.num_devices)
    local_bounds = tuple(
        _row_bounds(starts[rank], stops[rank]) for rank in range(dcp)
    )
    table, prefix_offsets, prefix_rows, n_pages = F.call_on_mesh(
        device_plan,
        mesh,
        # Per-device symbols (the table's columns, the prefix rows) and
        # per-rank values (the prefix offsets): ``Unknown``, not replicated,
        # or a subgraph call would type every device's value like device 0's.
        out_specs=[
            unknown,
            unknown,
            unknown,
            DeviceMapping(host_mesh, (Unknown(),) * mesh.ndim),
        ],
    )(
        input_row_offsets,
        kv_collection.cache_lengths,
        kv_collection.lookup_table,
        _per_device_host(batch_context_length, mesh.num_devices),
        _per_device_host(kv_collection.max_cache_length, mesh.num_devices),
        active,
        PerShard(range(mesh.num_devices)),
    )
    # One pages symbol for every device, the largest device's count: the
    # layers write the scratch buffers, and a subgraph call retypes a
    # written buffer argument from its layout, which keeps device 0's dims.
    most_pages = F.max(F.concat(n_pages.local_shards), axis=0)
    pages = _carrier(most_pages, f"{namespace}_pages").shape[0]

    # The query starts take the cache's batch symbol, as the table does.
    query_starts = F.call_on_mesh(
        lambda q, c: F.rebind(q, c.shape),
        mesh,
        out_specs=query_starts.mapping,
    )(query_starts, kv_collection.cache_lengths)
    active_rows = _carrier(
        F.reshape(F.cast(active, DType.int64), [1]), f"{namespace}_active"
    )
    plan = PrefillContextParallel(
        dcp=dcp,
        interleave=interleave,
        namespace=namespace,
        batch_row_offsets=input_row_offsets,
        row_offsets=row_offsets,
        query_starts=query_starts,
        row_start=row_start,
        row_stop=row_stop,
        rows=rows,
        restore=restore,
        zero_prefill_offsets=F.call_on_mesh(
            lambda o: F.broadcast_to(
                F.constant(0, DType.uint32, device=o.device),
                [1, o.shape[0]],
            ),
            mesh,
            out_specs=DeviceMapping.replicated(mesh),
        )(row_offsets),
        local_bounds=local_bounds,
        prefix_offsets=prefix_offsets,
        prefix_rows=prefix_rows,
        mla_scratch=_scratch_cache(kv_collection, pages, table),
        indexer_scratch=_scratch_cache(indexer_kv_collection, pages, table),
        active_rows=active_rows,
    )
    return dataclasses.replace(
        plan,
        mla_view_fields=plan.query_view_fields(plan.mla_scratch),
        indexer_view_fields=plan.query_view_fields(plan.indexer_scratch),
    )
