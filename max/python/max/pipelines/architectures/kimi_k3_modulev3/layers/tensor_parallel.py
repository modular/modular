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

from collections.abc import Sequence

from max import functional as F
from max.graph import ops
from max.nn.common_layers.mesh_axis import TP
from max.tensor import Tensor
from max.tensor.sharding import NamedMapping


def check_divisible(what: str, size: int, tp_size: int) -> None:
    """Raises if ``size`` does not split evenly across ``tp_size`` devices."""
    if size % tp_size:
        raise ValueError(
            f"{what} ({size}) is not divisible by the TP degree ({tp_size})."
        )


def fuse_rows(parts: Sequence[Tensor]) -> Tensor:
    """Concatenates weights along their first axis, device by device.

    Under tensor parallelism each device stacks its own shard of every
    row-sharded part and the whole of every replicated one, so one GEMM over
    the result hands each rank its own slice of every part. The parts are
    weights, so the graph compiler hoists the concat into initialization.
    """
    mesh = parts[0].mesh
    if TP not in mesh.axis_names or mesh.axis_size(TP) == 1:
        return F.concat(parts, axis=0)
    spec = (TP, *(None,) * (parts[0].rank - 1))
    return F.call_on_mesh(
        lambda *shards: ops.concat(shards, axis=0),
        mesh,
        out_specs=NamedMapping(mesh, spec),
    )(*parts)
