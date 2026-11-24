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
# Copied from
# max/python/max/pipelines/architectures/nemotron_h_modulev3/layers/sharding.py
# at main 4df8111afba.
"""Tensor-parallel placement of weights that stack several parts."""

from __future__ import annotations

from collections.abc import Sequence

from max import functional as F
from max.nn.common_layers.mesh_axis import TP
from max.nn.common_layers.quant_tensor import FP8BlockTensor, QTensor
from max.tensor import Tensor
from max.tensor.sharding import DeviceMesh, NamedMapping

__all__ = ["check_block_split", "check_segment_split", "shard_segments"]


def _sharded_parts(
    t: Tensor, segments: Sequence[int], rows: NamedMapping
) -> Tensor:
    """Shards each part of ``t``'s rows, keeping the parts in order."""
    return F.concat(
        [part.to(rows) for part in F.split(t, list(segments), axis=0)], axis=0
    )


def shard_segments(
    t: Tensor | QTensor, segments: Sequence[int], mesh: DeviceMesh
) -> Tensor | QTensor:
    """Shards each part of ``t``'s rows across ``mesh``'s tensor-parallel axis.

    ``t`` stacks parts of ``segments`` rows, and each device holds its share
    of every part, in order: ``concat`` along a sharded axis joins each
    device's pieces, which ``split`` by the same sizes takes apart again.
    ``t`` stays whole on the host as stored, so the graph compiler places
    it once, at model init.

    A block-scaled weight carries one scale row per ``block_size`` rows, so
    its scales take the split in blocks: each device's share of every
    segment must be a whole number of blocks.
    """
    if isinstance(t, FP8BlockTensor):
        block_m = t.block_size[0]
        check_segment_split(segments, block_m, mesh)
        rows = NamedMapping(mesh, (TP, None))
        return FP8BlockTensor(
            weight=_sharded_parts(t.weight, segments, rows),
            weight_scale_inv=_sharded_parts(
                t.weight_scale_inv,
                [segment // block_m for segment in segments],
                rows,
            ),
            block_size=t.block_size,
        )
    if not isinstance(t, Tensor):
        raise ValueError(
            f"{type(t).__name__} has no row-sharded layout here: Qwen3.5"
            " shards a plain weight or a block-scaled one."
        )
    rows = NamedMapping(mesh, (TP,) + (None,) * (t.rank - 1))
    return _sharded_parts(t, segments, rows)


def check_segment_split(
    segments: Sequence[int], block_m: int, mesh: DeviceMesh
) -> None:
    """Refuses a row split that cuts a stacked block-scaled weight in a block.

    :func:`shard_segments` gives each device ``segment / tp`` rows of every
    segment and ``segment / block_m / tp`` of its scale rows, which line up
    only when each device's share is a whole number of blocks.

    Args:
        segments: The row counts of the stacked parts.
        block_m: Rows per scale block.
        mesh: The mesh the weight is placed on.

    Raises:
        ValueError: If a device's share of a segment is not a whole number of
            blocks.
    """
    tp = mesh.axis_size(TP) if TP in mesh.axis_names else 1
    for segment in segments:
        if segment % tp or (segment // tp) % block_m:
            raise ValueError(
                f"cannot shard a {block_m}-row block-scaled weight across "
                f"{tp} devices in segments of {list(segments)} rows: each "
                f"device's {segment / tp:g} rows of the {segment}-row "
                "segment are not a whole number of blocks."
            )


def check_block_split(
    shape: Sequence[int],
    block_size: Sequence[int],
    sharding: Sequence[str | None],
    mesh: DeviceMesh,
) -> None:
    """Refuses a split that cuts a block-scaled weight inside a block.

    :meth:`FP8BlockTensor.to` splits the weight and its scales along the same
    axis, so each device's share of the weight lines up with its share of
    the scales only when that share is a whole number of blocks. ``(TP,
    None)`` splits the output rows and ``(None, TP)`` the input columns, the
    K axis the scales also block.

    Args:
        shape: The weight's ``[out_dim, in_dim]`` shape.
        block_size: Its ``(block_m, block_k)`` scale block.
        sharding: The weight's tensor-parallel spec, one entry per axis.
        mesh: The mesh it is placed on.

    Raises:
        ValueError: If an axis split across more than one device leaves a
            device a share that is not a whole number of blocks.
    """
    for axis, mesh_axis in enumerate(sharding):
        # A single-device mesh names no tensor-parallel axis, which leaves the
        # weight whole.
        if mesh_axis is None or mesh_axis not in mesh.axis_names:
            continue
        parts = mesh.axis_size(mesh_axis)
        if parts == 1:
            continue
        dim = int(shape[axis])
        block = int(block_size[axis])
        if dim % parts or (dim // parts) % block:
            raise ValueError(
                f"cannot split axis {axis} of a block-scaled weight of shape "
                f"{[int(d) for d in shape]} across {parts} devices: each "
                f"device's {dim / parts:g} of its {dim} entries are not a "
                f"whole number of {block}-wide blocks, so its scales would "
                "not line up with its weight."
            )
