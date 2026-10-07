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
"""Scalar promotion and a partial sum made from copies, checked against the
global value."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from max.driver import CPU
from max.experimental.sharding import (
    DeviceMapping,
    DeviceMesh,
    Partial,
    Placement,
    Replicated,
)
from max.experimental.tensor import Tensor

MESH = DeviceMesh((CPU(),) * 4, (2, 2), ("dp", "tp"))
R, P = Replicated(), Partial()
DATA = np.arange(6 * 8, dtype=np.float32).reshape(6, 8)


def _global(t: Tensor) -> npt.NDArray[np.float32]:
    whole = t.to(DeviceMapping.replicated(t.mesh))
    return np.asarray(
        np.from_dlpack(whole.local_shards[0].to(CPU()).driver_tensor),
        dtype=np.float32,
    )


def _placed(
    placements: tuple[Placement, ...],
) -> tuple[Tensor, npt.NDArray[np.float32]]:
    """``DATA`` placed as ``placements``, and its global value."""
    if any(isinstance(p, Partial) for p in placements):
        copies = 2 ** sum(isinstance(p, Partial) for p in placements)
        replicated = tuple(
            R if isinstance(p, Partial) else p for p in placements
        )
        base = Tensor(DATA).to(DeviceMapping(MESH, replicated))
        t = Tensor._from_shards(
            tuple(s.driver_tensor for s in base.local_shards),
            DeviceMapping(MESH, placements),
        )
        return t, DATA * copies
    return Tensor(DATA).to(DeviceMapping(MESH, placements)), DATA


def test_partial_plus_scalar() -> None:
    t, value = _placed((R, P))
    np.testing.assert_allclose(_global(t + 1.0), value + 1.0)


def test_replicated_partial_round_trip() -> None:
    t = Tensor(DATA).to(DeviceMapping(MESH, (R, R)))
    back = t.to(DeviceMapping(MESH, (P, P))).to(DeviceMapping(MESH, (R, R)))
    np.testing.assert_allclose(_global(back), DATA)
