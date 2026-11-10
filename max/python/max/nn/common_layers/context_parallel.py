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

DCP spans the attention tensor-parallel axis of the mesh (``dcp == tp``):
the whole of a one-axis mesh, or the ``tp`` axis of a ``(dp, tp)`` mesh of
data-parallel replicas, where every collective runs within a replica. The
``tp`` axis must be innermost, so a DCP group is a run of consecutive
devices and each collective is one launch over every replica. A smaller
DCP group inside TP needs a ``(tp / dcp, dcp)`` mesh; see
``mach/docs/internal/context-parallel-plan-2026-11-02.md``.
"""

from __future__ import annotations

import math

from max import functional as F
from max.driver import Device
from max.dtype import DType
from max.nn.common_layers.mesh_axis import TP, dcp_axis
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


def check_dcp_mesh(mesh: DeviceMesh, dcp: int) -> None:
    """Raises unless ``dcp`` spans the mesh's innermost attention TP axis.

    Raises:
        NotImplementedError: If ``dcp`` is not the size of the ``tp`` axis
            (the whole of a one-axis mesh), or another axis of more than one
            device is inside it.
    """
    axis = dcp_axis(mesh)
    inner = mesh.mesh_shape[axis + 1 :]
    if dcp != mesh.mesh_shape[axis] or any(size > 1 for size in inner):
        raise NotImplementedError(
            f"decode context parallelism over {dcp} devices needs it to span "
            f"the innermost {TP!r} axis of the mesh, got {mesh}; a group "
            "smaller than tp needs a (tp / dcp, dcp) mesh, which is not "
            "supported yet."
        )


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

    Every rank attended all ``H`` heads of its group over its own keys,
    producing ``out`` (``[T, H, D]``) and the per-head base-2 LSE ``lse``
    (``[T, H]``, float32). The merged output of head ``h`` is
    ``sum_r w_r * out_r`` with ``w_r = 2^(lse_r - log2(sum_s 2^lse_s))``,
    exactly attention over the union of the ranks' keys. Each rank sends
    every other rank its partials for that rank's heads in one all-to-all
    and merges what it receives (:func:`_combine_a2a`).

    Args:
        out: Each rank's partial attention output, ``[T, H, D]``.
        lse: Each rank's base-2 log-sum-exp per token and head, ``[T, H]``.
        dcp: The DCP group size, the size of the mesh's ``tp`` axis.

    Returns:
        The merged ``[T, H, D]`` output, sharded over its head axis along
        the DCP axis, each rank holding its ``H / dcp`` heads, ready for a
        head-sharded output projection. Other mesh axes keep ``out``'s
        placements: each data-parallel replica merges only its own ranks.
    """
    check_dcp_mesh(out.mesh, dcp)
    return _combine_a2a(out, lse, dcp)


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

    Each rank packs its partial outputs and LSEs per destination rank
    (:func:`_pack_partials`); the all-to-all splits that axis, so rank
    ``j`` receives every rank's partials for heads
    ``[j * H / dcp, (j + 1) * H / dcp)``, and merges them locally
    (:func:`_merge_received`). One collective per layer, where an LSE
    all-gather plus an output reduce-scatter would take two.
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
    return F.call_on_mesh(
        lambda received, partial: _merge_received(received, partial, dcp),
        mesh,
        out_specs=_on_dcp_axis(out, Sharded(1)),
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

    Rank ``r``'s columns land at ``[r * c, (r + 1) * c)``, so a tensor whose
    columns are a rank's run of heads gathers into all of them in order. On
    a ``(dp, tp)`` mesh each replica gathers only its own ranks' columns.
    """
    mesh = x.mesh
    check_dcp_mesh(mesh, dcp)
    axis = dcp_axis(mesh)
    per_rank = _on_dcp_axis(x, Sharded(1), Unknown())
    gathered = F.allgather(
        x.rebind_mapping(per_rank), tensor_axis=1, mesh_axis=axis
    )
    # Back to the input's placements on the other axes.
    return gathered.rebind_mapping(_on_dcp_axis(x, gathered.placements[axis]))
