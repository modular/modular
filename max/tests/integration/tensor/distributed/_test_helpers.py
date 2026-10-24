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
"""Shared test helpers for distributed tensor tests."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from max.experimental.functional import transfer_to
from max.experimental.sharding import (
    DeviceMapping,
    DeviceMesh,
    Partial,
    Placement,
    Replicated,
)
from max.experimental.tensor import Tensor


def make_partial(
    data: np.ndarray[Any, Any],
    mesh: DeviceMesh,
    placements: tuple[Placement, ...],
) -> Tensor:
    """Creates a Partial tensor by replicating data across all devices.

    Distributes with Replicated placement (full copy on each device),
    then re-labels as the requested Partial placement.
    """
    replicated = DeviceMapping(mesh, tuple(Replicated() for _ in placements))
    t = transfer_to(Tensor(data), replicated)
    return Tensor._from_shards(
        tuple(s.driver_tensor for s in t.local_shards),
        DeviceMapping(mesh, placements),
    )


def assert_no_shadowed_tests(cls: type) -> None:
    """Fails if a ``test_*`` name is defined by more than one class in ``cls``.

    Aggregators such as ``ReductionTests`` inherit one mixin per op. Python
    keeps only the first definition of a name along the MRO, so a second mixin
    that reuses ``test_non_sharded_axis`` is silently never run. This reads each
    class's own namespace instead of ``getattr``, which only sees the winner.

    Args:
        cls: The aggregator class to check.

    Raises:
        AssertionError: If any test name has more than one defining class.
    """
    owners: dict[str, list[str]] = {}
    for klass in cls.__mro__:
        for name in vars(klass):
            if name.startswith("test_"):
                owners.setdefault(name, []).append(klass.__name__)
    hidden = {name: who for name, who in owners.items() if len(who) > 1}
    assert not hidden, (
        f"{cls.__name__}: these tests are defined in more than one base, so"
        " only the first (listed first) runs; give each a unique name: "
        + "; ".join(
            f"{name} in {', '.join(who)}" for name, who in hidden.items()
        )
    )


def make_partial_distinct(
    fragment_fn: Callable[[int], np.ndarray[Any, Any]],
    mesh: DeviceMesh,
) -> Tensor:
    """Creates a 1-axis Partial tensor whose per-device fragments differ.

    ``make_partial`` copies identical data to every device, which hides
    per-rank identity: an op that sums fragments looks the same as one
    that mishandles them. This places each device's own fragment from
    ``fragment_fn(rank)`` on that device and labels the set Partial: the
    tensor's value is the sum of the fragments.
    """
    if mesh.ndim != 1:
        raise ValueError(
            f"make_partial_distinct needs a 1-axis mesh; got {mesh.ndim}."
        )
    shards = []
    for rank, device in enumerate(mesh.devices):
        fragment = Tensor(fragment_fn(rank))
        if fragment.device != device:
            fragment = Tensor(storage=fragment.driver_tensor.to(device))
        shards.append(fragment.driver_tensor)
    return Tensor._from_shards(tuple(shards), DeviceMapping(mesh, (Partial(),)))
