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
"""Tensor-parallel placement helpers for Nemotron-H layers."""

from __future__ import annotations

from collections.abc import Sequence

from max.experimental import functional as F
from max.experimental.nn.common_layers.mesh_axis import TP
from max.experimental.sharding import DeviceMesh, NamedMapping
from max.experimental.tensor import Tensor


def shard_axis(t: Tensor, axis: int = 0) -> Tensor:
    """Shards ``t`` on ``axis`` across its mesh's tensor-parallel axis."""
    spec: list[str | None] = [None] * t.rank
    spec[axis] = TP
    return t.to(NamedMapping(t.mesh, tuple(spec)))


def shard_segments(
    t: Tensor, segments: Sequence[int], mesh: DeviceMesh
) -> Tensor:
    """Shards each part of ``t``'s rows across ``mesh``'s tensor-parallel axis.

    ``t`` stacks parts of ``segments`` rows, and each device holds its share
    of every part, in order: ``concat`` along a sharded axis joins each
    device's pieces, which ``split`` by the same sizes takes apart again.
    ``t`` stays whole on the host as stored, so the graph compiler places
    it once, at model init.
    """
    rows = NamedMapping(mesh, (TP,) + (None,) * (t.rank - 1))
    return F.concat(
        [part.to(rows) for part in F.split(t, list(segments), axis=0)], axis=0
    )
