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
"""Tensor-parallel sharding helpers shared by Kimi K3's layers."""

from __future__ import annotations

from max.experimental.nn.common_layers.mesh_axis import TP
from max.experimental.sharding import DeviceMesh, NamedMapping
from max.experimental.tensor import Tensor


def check_divisible(what: str, size: int, tp_size: int) -> None:
    """Raises if ``size`` does not split evenly across ``tp_size`` devices."""
    if size % tp_size:
        raise ValueError(
            f"{what} ({size}) is not divisible by the TP degree ({tp_size})."
        )


def shard_fused_blocks(
    weight: Tensor, mesh: DeviceMesh, num_blocks: int, axis: int
) -> Tensor:
    """Shards a weight that fuses ``num_blocks`` equal blocks along ``axis``.

    A contiguous shard of the fused axis would give one device whole blocks
    and another none. This reorders the axis from block-major to
    device-major first, so each device's contiguous shard holds its slice of
    every block.
    """
    tp_size = mesh.axis_size(TP)
    shape = [int(dim) for dim in weight.shape]
    block = shape[axis] // num_blocks
    check_divisible("Fused block width", block, tp_size)
    split = (
        shape[:axis]
        + [num_blocks, tp_size, block // tp_size]
        + shape[axis + 1 :]
    )
    reordered = weight.reshape(split).transpose(axis, axis + 1).reshape(shape)
    spec = tuple(TP if dim == axis else None for dim in range(len(shape)))
    return reordered.to(NamedMapping(mesh, spec))
