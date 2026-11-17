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
"""Decode context parallelism (DCP) helpers shared by attention layers.

Under DCP the ranks of an attention tensor-parallel group stop replicating the
KV cache: each stores only the token positions it owns (see the Mojo
``kv_cache.context_parallel`` map), attends every head of the group over its
share, and the partial results merge exactly by their log-sum-exp (LSE), the
same correction split-K attention applies across thread blocks.

Nothing here is model-specific. A layer needs three things: each device's
DCP rank (:func:`dcp_ranks`), the merge of its partial attention back to its
own heads (:func:`combine_partial_attention`), and, for an indexer that picks
the keys to attend, the exchange of its local top-k scores
(:func:`gather_topk_scores`).

A DCP group is ``dcp`` consecutive ranks of the attention tensor-parallel
axis of the mesh: ``tp`` of a one-axis mesh or of a ``(dp, tp)`` mesh of
data-parallel replicas, which must be innermost. ``dcp`` may be any divisor
of ``tp``. With ``dcp < tp`` the ``tp / dcp`` groups each hold a full copy
of the cache between their ranks, and each attends its own run of
``H * dcp / tp`` heads, so the group's merge leaves every rank exactly its
``H / tp`` tensor-parallel heads.

The model keeps its mesh. What only a group sees runs on the mesh's DCP
view (:func:`dcp_view`), the same devices with ``tp`` split into
``(tp / dcp, dcp)``: the collectives over one group, which run along its
``dcp`` axis, and the weights a group shares, which shard along its ``tp``
axis. :func:`to_dcp_mesh` moves a tensor between the two meshes without
moving data. A placement only the view can state, such as a tensor sharded
across groups and replicated within one, is ``Unknown`` on the model mesh.
"""

from __future__ import annotations

import math

from max import functional as F
from max.driver import Device
from max.dtype import DType
from max.nn.common_layers.mesh_axis import DCP, TP, dcp_axis
from max.nn.kernels import cp_merge_topk
from max.tensor import Tensor
from max.tensor.sharding import (
    DeviceMapping,
    DeviceMesh,
    Placement,
    Sharded,
    Unknown,
)
from max.tensor.sharding.action import PerShard

_LN2 = math.log(2.0)
# A finite stand-in for the ``-inf`` LSE of a rank with no keys. Weights stay
# exactly 0 next to any real LSE, and a row no rank has keys for merges to 0
# instead of NaN (``-inf - -inf``).
_LSE_FLOOR = -1e30

__all__ = [
    "check_dcp_mesh",
    "combine_partial_attention",
    "dcp_group_heads",
    "dcp_rank_tensor",
    "dcp_ranks",
    "dcp_view",
    "gather_columns",
    "gather_topk_scores",
    "merge_topk",
    "split_selection",
    "to_dcp_mesh",
]


def check_dcp_mesh(mesh: DeviceMesh, dcp: int) -> None:
    """Raises unless DCP groups of ``dcp`` ranks tile the attention TP axis.

    The axis is the innermost ``tp`` axis, the whole of a one-axis mesh, or
    the ``dcp`` axis of a DCP view (:func:`dcp_view`).

    Raises:
        NotImplementedError: If ``dcp`` does not divide the size of the
            ``tp`` axis (or is not the size of a view's ``dcp`` axis), or
            another axis of more than one device is inside it.
    """
    axis = dcp_axis(mesh)
    size = mesh.mesh_shape[axis]
    inner = mesh.mesh_shape[axis + 1 :]
    fits = size == dcp if DCP in mesh.axis_names else size % dcp == 0
    if dcp < 1 or not fits or any(n > 1 for n in inner):
        raise NotImplementedError(
            f"decode context parallelism over {dcp} devices needs groups of "
            f"{dcp} consecutive ranks of the innermost {TP!r} axis of the "
            f"mesh, so dcp must divide its size; got {mesh}."
        )


def dcp_view(mesh: DeviceMesh, dcp: int) -> DeviceMesh:
    """Returns ``mesh`` with its attention TP axis split into DCP groups.

    The view holds the same devices in the same order, with the ``tp`` axis
    (:func:`~max.nn.common_layers.mesh_axis.dcp_axis`) split into a ``tp``
    axis of ``tp / dcp`` groups and an innermost ``dcp`` axis of each
    group's ranks: ``(dp, tp)`` becomes ``(dp, tp / dcp, dcp)``. A
    collective along ``dcp`` runs within each group, and a weight sharded
    along the view's ``tp`` axis and replicated along ``dcp`` gives each
    group its own heads. With ``dcp == tp`` the view's ``tp`` axis has one
    group. A view is its own view.

    Raises:
        NotImplementedError: If ``dcp`` does not tile the mesh's attention
            TP axis (see :func:`check_dcp_mesh`).
    """
    check_dcp_mesh(mesh, dcp)
    if DCP in mesh.axis_names:
        return mesh
    axis = dcp_axis(mesh)
    groups = mesh.mesh_shape[axis] // dcp
    return DeviceMesh(
        mesh.devices,
        mesh.mesh_shape[:axis] + (groups, dcp) + mesh.mesh_shape[axis + 1 :],
        mesh.axis_names[:axis] + (TP, DCP) + mesh.axis_names[axis + 1 :],
    )


def _view_placements(
    placements: tuple[Placement, ...], axis: int
) -> tuple[Placement, ...]:
    """Returns a model-mesh mapping's placements on its DCP view.

    The TP axis at ``axis`` becomes the view's ``(tp, dcp)`` pair. Its
    placement holds on both, since the pair spans the same devices in the
    same row-major order: a tensor axis sharded over ``tp`` is sharded over
    both, the groups' pieces first.
    """
    p = placements[axis]
    return placements[:axis] + (p, p) + placements[axis + 1 :]


def _model_placements(
    placements: tuple[Placement, ...], axis: int, sizes: tuple[int, int]
) -> tuple[Placement, ...]:
    """Returns a view mapping's placements on the mesh it views.

    ``axis`` is the view's ``tp`` axis and ``sizes`` the sizes of its
    ``(tp, dcp)`` pair. The pair folds into one placement when both say the
    same, or when one of them spans a single device. Anything else, such as
    sharded across groups and replicated within one, has no placement on
    the model mesh and becomes ``Unknown``.
    """
    outer, inner = placements[axis], placements[axis + 1]
    if sizes[0] == 1:
        p = inner
    elif sizes[1] == 1 or outer == inner:
        p = outer
    else:
        p = Unknown()
    return placements[:axis] + (p,) + placements[axis + 2 :]


def to_dcp_mesh(t: Tensor, mesh: DeviceMesh) -> Tensor:
    """Moves ``t`` between a mesh and its DCP view, moving no data.

    One of ``t.mesh`` and ``mesh`` must be the other's :func:`dcp_view`.
    Each device keeps its shard, and the placements carry over (see
    :func:`_view_placements` and :func:`_model_placements`). A tensor on
    ``mesh`` already, or on one device, is returned as it is.

    Raises:
        ValueError: If neither mesh is the other's DCP view.
    """
    source = t.mesh
    if not t.is_distributed or source == mesh:
        return t
    if source.devices != mesh.devices:
        raise ValueError(f"{mesh} is not a DCP view of {source}, or back")
    if DCP in mesh.axis_names and DCP not in source.axis_names:
        axis = dcp_axis(source)
        placements = _view_placements(t.placements, axis)
    elif DCP in source.axis_names and DCP not in mesh.axis_names:
        axis = source.axis_names.index(TP)
        sizes = (source.mesh_shape[axis], source.mesh_shape[axis + 1])
        placements = _model_placements(t.placements, axis, sizes)
    else:
        raise ValueError(f"{mesh} is not a DCP view of {source}, or back")
    return F.call_on_mesh(
        lambda shard: shard,
        source,
        out_specs=DeviceMapping(mesh, placements),
    )(t)


def dcp_group_heads(mesh: DeviceMesh, n_heads: int, dcp: int) -> int:
    """Returns the query heads each DCP group of ``mesh`` attends.

    A group attends its ranks' tensor-parallel heads: ``n_heads * dcp /
    tp``, all of them when ``dcp == tp``.
    """
    check_dcp_mesh(mesh, dcp)
    tp = mesh.axis_size(dcp_axis(mesh))
    if DCP in mesh.axis_names:
        tp *= mesh.axis_size(TP)
    return n_heads * dcp // tp


def dcp_ranks(mesh: DeviceMesh, dcp: int) -> PerShard[int]:
    """Returns each device's rank in its DCP group, for ``call_on_mesh``.

    A device's rank is its coordinate along the DCP axis
    (:func:`dcp_axis`). That axis is innermost, so DCP groups are runs of
    ``dcp`` consecutive devices in the mesh's row-major order, and device
    ``i`` is rank ``i % dcp``.
    """
    axis = dcp_axis(mesh)
    return PerShard(
        mesh.device_coord(i, axis) % dcp for i in range(mesh.num_devices)
    )


def _on_dcp_axis(
    t: Tensor, placement: Placement, others: Placement | None = None
) -> DeviceMapping:
    """Returns ``t``'s mapping with ``placement`` on the DCP axis.

    The other axes, data-parallel replicas when there are any, keep ``t``'s
    placements, or take ``others`` when given.
    """
    mesh = t.mesh
    axis = dcp_axis(mesh)
    placements = [
        placement if i == axis else (p if others is None else others)
        for i, p in enumerate(t.placements)
    ]
    return DeviceMapping(mesh, tuple(placements))


def dcp_rank_tensor(mesh: DeviceMesh, dcp: int) -> Tensor:
    """Returns each device's DCP rank as an ``int32`` ``[1]`` device tensor.

    The indexer and its top-k merge read the rank on the device; the KV
    stores take it as a host scalar instead (:func:`dcp_ranks`).
    """

    def _rank(rank: int, device: Device) -> Tensor:
        return F.constant([rank], DType.int32, device=device)

    return F.call_on_mesh(_rank, mesh)(
        dcp_ranks(mesh, dcp), PerShard(mesh.devices)
    )


def _pack_selection(rows: Tensor, lengths: Tensor) -> Tensor:
    return F.concat([rows, F.unsqueeze(lengths, 1)], axis=1)


def _unpack_selection(selection: Tensor, width: int) -> tuple[Tensor, Tensor]:
    return selection[:, :width], selection[:, width]


def merge_topk(local_rows: Tensor, local_scores: Tensor, dcp: int) -> Tensor:
    """Returns each rank's share of the group-wide top-k as one selection.

    Each rank picked the top ``k`` of its own keys (``local_rows``, its
    physical rows, and ``local_scores``). Any key in the group's top ``k``
    is in its owner's local top ``k``, so merging the ``dcp * k`` gathered
    scores is exact up to ties. The result packs each token's owned rows,
    compacted to the front, with their count in a last column: an int32
    ``[T, k + 1]`` selection (see :func:`split_selection`). The packed form
    lets ``shared`` indexer layers reuse it like a plain top-k.

    Over an NVFP4 MLA cache ``local_rows`` is ``[T, 2 * k]``, each
    selection's value row then its scale-pool row, and the selection is
    ``[T, 2 * k + 1]``: both halves compacted the same way, then the count.
    """
    mesh = local_rows.mesh
    k = int(local_scores.shape[1])
    halves = int(local_rows.shape[1]) // k
    gathered = gather_topk_scores(local_scores, dcp)
    rows, lengths = cp_merge_topk(
        local_rows, gathered, dcp_rank_tensor(mesh, dcp), dcp, halves
    )
    return F.call_on_mesh(_pack_selection, mesh)(rows, lengths)


def split_selection(selection: Tensor, width: int) -> tuple[Tensor, Tensor]:
    """Splits a packed selection into its rows and ``[T]`` counts.

    ``width`` is the rows' width: the top-k, or twice it over an NVFP4 MLA
    cache, whose rows are the value rows then the scale-pool rows.
    """
    return F.call_on_mesh(_unpack_selection, selection.mesh)(selection, width)


def combine_partial_attention(out: Tensor, lse: Tensor, dcp: int) -> Tensor:
    """Merges each rank's partial attention and returns its own heads.

    Every rank attended all ``H`` heads of its DCP group over its own keys,
    producing ``out`` (``[T, H, D]``) and the per-head base-2 LSE ``lse``
    (``[T, H]``, float32). The merged output of head ``h`` is
    ``sum_r w_r * out_r`` with ``w_r = 2^(lse_r - log2(sum_s 2^lse_s))``,
    exactly attention over the union of the ranks' keys. Each rank sends
    every other rank of its group its partials for that rank's heads in one
    all-to-all and merges what it receives (:func:`_combine_a2a`).

    Args:
        out: Each rank's partial attention output, ``[T, H, D]``, over its
            group's ``H`` heads: all of them when ``dcp`` is the size of
            the ``tp`` axis, else the group's run of them
            (:func:`dcp_group_heads`).
        lse: Each rank's base-2 log-sum-exp per token and head, ``[T, H]``.
        dcp: The DCP group size, a divisor of the ``tp`` axis's size.

    Returns:
        The merged output on ``out``'s mesh, sharded over its head axis
        along the ``tp`` axis: each rank holds its ``H / dcp`` heads of its
        group's, which are its tensor-parallel heads, ready for a
        head-sharded output projection. Other mesh axes keep ``out``'s
        placements: each data-parallel replica merges only its own ranks.
    """
    mesh = out.mesh
    view = dcp_view(mesh, dcp)
    merged = _combine_a2a(to_dcp_mesh(out, view), to_dcp_mesh(lse, view), dcp)
    return to_dcp_mesh(merged, mesh)


def _lse_lanes(lse: Tensor, dtype: DType) -> list[Tensor]:
    """Encodes a float32 LSE exactly as ``dtype`` lanes beside the output.

    The graph has no bitcast, so a bfloat16 output carries each LSE as three
    bfloat16 lanes, ``hi + mid + lo``: each lane takes the next 8 of
    float32's 24 significand bits, each residual is exact in float32, and
    the lanes sum back to the LSE exactly in any order. A float32 output
    carries it as itself.
    """
    if dtype == DType.float32:
        return [lse]
    if dtype != DType.bfloat16:
        raise NotImplementedError(
            f"the a2a DCP merge packs bfloat16 or float32 outputs, got {dtype}"
        )
    lanes = []
    rest = lse
    for _ in range(3):
        lane = F.cast(rest, dtype)
        lanes.append(lane)
        rest = rest - F.cast(lane, DType.float32)
    return lanes


def _pack_partials(out: Tensor, lse: Tensor, dcp: int) -> Tensor:
    """Packs one rank's partials into one chunk per destination rank.

    Returns ``[T, dcp, (D + lanes) * h]`` in the output's dtype, ``h = H /
    dcp``: chunk ``r`` holds destination ``r``'s heads as ``[D + lanes, h]``
    rows, the output's ``D`` features and then the LSE lanes
    (:func:`_lse_lanes`), each row one value per head. Splitting axis 1
    gives each destination its heads' partials in one piece.

    Head-minor rows keep the pack a concat of whole rows: the output moves
    as row vectors and each lane is one more row. Appending 1-wide lanes to
    every head's 512-wide row instead left the concat one element per
    thread: 19 us per layer at batch 64 on B200.
    """
    lse = F.cast(lse, DType.float32)
    # A rank with no keys returns a -inf LSE and zero output; the floor gives
    # it weight 0, or an equal share of zeros when no rank has keys, and
    # keeps the lane split finite.
    lse = F.max(lse, F.constant(_LSE_FLOOR, DType.float32, device=lse.device))
    tokens = out.shape[0]
    heads, depth = int(out.shape[1]), int(out.shape[2])
    h = heads // dcp
    rows = F.permute(F.reshape(out, [tokens, dcp, h, depth]), [0, 1, 3, 2])
    lanes = [
        F.reshape(lane, [tokens, dcp, 1, h])
        for lane in _lse_lanes(lse, out.dtype)
    ]
    packed = F.concat([rows, *lanes], axis=2)
    return F.reshape(packed, [tokens, dcp, (depth + len(lanes)) * h])


def _merge_received(received: Tensor, partial: Tensor, dcp: int) -> Tensor:
    """Merges the ``dcp`` partials of this rank's heads.

    ``received`` is ``[T, dcp, (D + lanes) * h]``: chunk ``s`` holds source
    rank ``s``'s ``[D + lanes, h]`` partials for this rank's ``h = H / dcp``
    heads (:func:`_pack_partials`). ``partial`` is this rank's own
    ``[T, H, D]`` output, for ``H``, ``D`` and the dtype. Returns
    ``[T, h, D]``.
    """
    tokens = received.shape[0]
    depth = int(partial.shape[2])
    heads = int(partial.shape[1]) // dcp
    width = int(received.shape[2]) // heads
    split = F.reshape(received, [tokens, dcp, width, heads])
    outs = F.cast(split[:, :, :depth, :], DType.float32)
    lanes = F.cast(split[:, :, depth:, :], DType.float32)
    lse = F.sum(lanes, axis=2)  # [T, dcp, 1, h]
    # Max-shift before exponentiating.
    peak = F.max(lse, axis=1)  # [T, 1, 1, h]
    weight = F.exp((lse - peak) * _LN2)
    weight = weight / F.sum(weight, axis=1)
    merged = F.squeeze(F.sum(outs * weight, axis=1), 1)  # [T, D, h]
    return F.cast(F.permute(merged, [0, 2, 1]), partial.dtype)


def _combine_a2a(out: Tensor, lse: Tensor, dcp: int) -> Tensor:
    """Merges by one packed all-to-all, as vLLM's and SGLang's a2a do.

    ``out`` and ``lse`` are on a DCP view (:func:`dcp_view`). Each rank
    packs its partial outputs and LSEs per destination rank
    (:func:`_pack_partials`); the all-to-all along the ``dcp`` axis splits
    that axis, so rank ``j`` of a group receives every rank's partials for
    the group's heads ``[j * H / dcp, (j + 1) * H / dcp)``, and merges them
    locally (:func:`_merge_received`). One collective per layer, where an
    LSE all-gather plus an output reduce-scatter would take two.
    """
    mesh = out.mesh
    axis = dcp_axis(mesh)
    # The packed partials claim nothing on the other axes: a replica's
    # tokens are its own.
    packed = F.call_on_mesh(
        lambda out, lse: _pack_partials(out, lse, dcp),
        mesh,
        out_specs=_on_dcp_axis(out, Unknown(), Unknown()),
    )(out, lse)
    # The packed partials are Unknown on the DCP axis, so the exchange
    # claims nothing either: each rank holds every rank's partials for its
    # heads, not a slice of one tensor.
    received = F.all_to_all(packed, split_axis=1, concat_axis=1, mesh_axis=axis)
    # Group ``g`` attended heads ``[g * H, (g + 1) * H)`` of the layer's, so
    # each rank's merged heads are its slice along the whole TP axis: the
    # view's ``tp`` axis, then ``dcp``.
    groups = mesh.axis_names.index(TP)
    merged = DeviceMapping(
        mesh,
        tuple(
            Sharded(1) if i in (groups, axis) else p
            for i, p in enumerate(out.placements)
        ),
    )
    return F.call_on_mesh(
        lambda received, partial: _merge_received(received, partial, dcp),
        mesh,
        out_specs=merged,
    )(received, out)


def gather_topk_scores(scores: Tensor, dcp: int) -> Tensor:
    """All-gathers each rank's local top-k scores, rank-major.

    Rank ``r``'s ``k`` scores land in columns ``[r * k, (r + 1) * k)`` of
    the ``[T, dcp * k]`` result. Only scores travel: a column names its
    source rank, and each rank keeps its own rows. On a ``(dp, tp)`` mesh
    each replica gathers only its own ranks' scores.
    """
    return gather_columns(scores, dcp)


def gather_columns(x: Tensor, dcp: int) -> Tensor:
    """All-gathers each rank's ``[T, c]`` columns into ``[T, dcp * c]``.

    Rank ``r`` of a DCP group gets its columns at ``[r * c, (r + 1) * c)``,
    so a tensor whose columns are a rank's run of heads gathers into all of
    its group's heads in order. Each group gathers only its own ranks'
    columns, and on a ``(dp, tp)`` mesh only within its replica. The result
    is on ``x``'s mesh, ``Replicated`` along ``tp`` when one group spans
    it, else ``Unknown`` there: the groups gathered different columns.
    """
    mesh = x.mesh
    view = dcp_view(mesh, dcp)
    x = to_dcp_mesh(x, view)
    axis = dcp_axis(view)
    per_rank = _on_dcp_axis(x, Sharded(1), Unknown())
    gathered = F.allgather(
        x.rebind_mapping(per_rank), tensor_axis=1, mesh_axis=axis
    )
    # Back to the input's placements on the other axes.
    gathered = gathered.rebind_mapping(
        _on_dcp_axis(x, gathered.placements[axis])
    )
    return to_dcp_mesh(gathered, mesh)
