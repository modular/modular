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

from max.nn.common_layers.mesh_axis import TP
from max.tensor import Tensor
from max.tensor.sharding import DeviceMesh, NamedMapping


def check_divisible(what: str, size: int, tp_size: int) -> None:
    """Raises if ``size`` does not split evenly across ``tp_size`` devices."""
    if size % tp_size:
        raise ValueError(
            f"{what} ({size}) is not divisible by the TP degree ({tp_size})."
        )


def shard_interleaved_blocks(
    weight: Tensor, mesh: DeviceMesh, num_blocks: int, axis: int
) -> Tensor:
    """Shards a weight that fuses ``num_blocks`` equal blocks along ``axis``.

    A contiguous shard of a block-major fused axis would give one device
    whole blocks and another none, so the weight must already be interleaved
    per TP rank -- each rank's contiguous slice holding its slice of every
    block -- as the weight adapter's ``interleave_for_tp`` writes it. This
    only checks the block width splits evenly and shards contiguously: a
    reorder here would act on the lazily built placeholder, not on the
    checkpoint arrays the loader splits on the host.
    """
    tp_size = mesh.axis_size(TP)
    shape = [int(dim) for dim in weight.shape]
    check_divisible("Fused block width", shape[axis] // num_blocks, tp_size)
    spec = tuple(TP if dim == axis else None for dim in range(len(shape)))
    return weight.to(NamedMapping(mesh, spec))
