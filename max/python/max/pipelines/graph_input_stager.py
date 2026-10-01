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

"""Staging for the graph inputs a forward writes each step."""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field

from max.driver import (
    Buffer,
    Device,
    Usage,
    batch_inplace_copy,
)
from max.dtype import DType

_ALIGNMENT = 256
"""Byte boundary every arena region starts on.

It matches what a fresh allocation gets, and ``Buffer.view`` does not check
alignment, so the arena has to keep it.
"""

_FUSE_MAX_BYTES = 64 * 1024
"""Inputs wider than this get an arena of their own.

A shared arena's host staging is allocated in full every step, so a wide
member would make every step pay for its capacity.
"""


@dataclass(frozen=True)
class InputDescriptor:
    """One graph input: its name, its largest shape, and where it goes."""

    name: str
    """Prefix it with the replica index for anything a replica owns."""

    dtype: DType

    max_shape: tuple[int, ...]
    """The shape at the pipeline's batching limits. Staging more raises."""

    destinations: Sequence[Device]
    """Devices the input is copied to. Host staging is allocated on the
    first."""

    @property
    def nbytes(self) -> int:
        """The input's size in bytes at :attr:`max_shape`."""
        return math.prod(self.max_shape) * self.dtype.size_in_bytes


@dataclass(eq=False)
class _Arena:
    """Inputs sharing a destination list, each at a fixed offset.

    Members staged next to each other go out in one copy per destination
    instead of one copy each. The device buffers are allocated once and kept,
    since captured graphs bind them.
    """

    destinations: tuple[Device, ...]
    size: int = 0
    members: int = 0
    _device: tuple[Buffer, ...] | None = field(
        default=None, init=False, repr=False
    )

    def device_buffers(self) -> tuple[Buffer, ...]:
        """Returns the per-destination buffers, allocating them on first use."""
        if self._device is None:
            self._device = tuple(
                Buffer(shape=(self.size,), dtype=DType.uint8, device=device)
                for device in self.destinations
            )
        return self._device


@dataclass(frozen=True)
class _Slot:
    """Where one input lives in its arena."""

    descriptor: InputDescriptor
    arena: _Arena
    offset: int
    region_end: int
    """Where the next member's region starts."""


class GraphInputStaging:
    """One forward's staging, handed out by :meth:`GraphInputStager.stage`.

    Inputs requested in the scope are sent when it closes. The rest keep the
    value the graph last read.
    """

    def __init__(self, slots: Mapping[str, _Slot]) -> None:
        self._slots = slots
        self._staged: dict[str, tuple[Buffer, tuple[Buffer, ...]]] = {}
        self._hosts: dict[_Arena, Buffer] = {}
        self._ranges: dict[_Arena, list[tuple[int, int, int]]] = {}

    def get(
        self, name: str, shape: tuple[int, ...]
    ) -> tuple[Buffer, tuple[Buffer, ...]]:
        """Returns the host staging for ``name`` and its device destinations.

        The host buffer is new every step: with the overlap scheduler, a
        previous step's copy may still be reading the old one, so host buffers
        can't be recycled.

        Raises:
            KeyError: If ``name`` was never described.
            RuntimeError: If ``shape`` outgrows the described maximum, or
                disagrees with an earlier call in this scope.
        """
        shape = tuple(shape)
        staged = self._staged.get(name)
        if staged is not None:
            # The shards of a replica share metadata, so they all ask for it.
            if tuple(staged[0].shape) != shape:
                raise RuntimeError(
                    f"Graph input {name!r} was staged as "
                    f"{list(staged[0].shape)} and again as {list(shape)} in "
                    "one forward"
                )
            return staged

        slot = self._slots[name]
        descriptor, arena = slot.descriptor, slot.arena
        num_elements = math.prod(shape)
        if num_elements > math.prod(descriptor.max_shape):
            raise RuntimeError(
                f"Graph input {name!r} needs {num_elements} elements "
                f"(shape={list(shape)}), more than its "
                f"max_shape={list(descriptor.max_shape)} allows."
            )
        end = slot.offset + num_elements * descriptor.dtype.size_in_bytes

        if arena not in self._hosts:
            # A shared arena doesn't know yet which members this step will
            # stage, so its host staging covers the whole region.
            self._hosts[arena] = Buffer(
                shape=(arena.size if arena.members > 1 else end,),
                dtype=DType.uint8,
                device=arena.destinations[0],
                usage=Usage.STAGING | Usage.UNTRACKED,
            )
        self._ranges.setdefault(arena, []).append(
            (slot.offset, end, slot.region_end)
        )

        host, *devices = (
            buffer[slot.offset : end].view(descriptor.dtype, shape)
            for buffer in (self._hosts[arena], *arena.device_buffers())
        )
        self._staged[name] = (host, tuple(devices))
        return self._staged[name]

    def send(self) -> None:
        """Copies everything staged in this scope, one submission per device.

        Called by :meth:`GraphInputStager.stage` on the way out.
        """
        dsts: list[Buffer] = []
        srcs: list[Buffer] = []
        for arena, ranges in self._ranges.items():
            # Ranges merge while each starts where the previous member's region
            # ends. A skipped member breaks the run, since that part of the new
            # host buffer was never written.
            runs: list[list[int]] = []
            for start, end, region_end in sorted(ranges):
                if runs and runs[-1][2] == start:
                    runs[-1][1:] = [end, region_end]
                else:
                    runs.append([start, end, region_end])
            host = self._hosts[arena]
            for start, end, _ in runs:
                for device_buffer in arena.device_buffers():
                    dsts.append(device_buffer[start:end])
                    srcs.append(host[start:end])
        batch_inplace_copy(dsts, srcs)


class GraphInputStager:
    """Device input buffers sized to the largest batch the pipeline allows.

    A step opens a :meth:`stage` scope, asks for the inputs it needs, fills
    the host side, and everything it staged is copied when the scope closes.
    Small inputs with the same destinations share an arena, so a step pays
    the per-copy overhead once per destination rather than once per input.

    One instance serves every KV leaf, tensor-parallel shard and data-parallel
    replica. Replicas need a name prefix because on CPU every replica gets the
    same ``Device``.

    Args:
        inputs: Every input a forward can stage. A name appears once.

    Raises:
        ValueError: If a name repeats, or an input has no destination.
    """

    def __init__(self, inputs: Iterable[InputDescriptor]) -> None:
        descriptors: dict[str, InputDescriptor] = {}
        for descriptor in inputs:
            if not descriptor.destinations:
                raise ValueError(
                    f"Graph input {descriptor.name!r} needs at least one "
                    "destination device"
                )
            if descriptor.name in descriptors:
                raise ValueError(
                    f"Graph input {descriptor.name!r} is described twice"
                )
            descriptors[descriptor.name] = descriptor

        # Smallest first, so a run cut short carries less padding.
        shared: dict[tuple[Device, ...], _Arena] = {}
        self._slots: dict[str, _Slot] = {}
        for descriptor in sorted(descriptors.values(), key=lambda d: d.nbytes):
            destinations = tuple(descriptor.destinations)
            arena = (
                _Arena(destinations)
                if descriptor.nbytes > _FUSE_MAX_BYTES
                else shared.setdefault(destinations, _Arena(destinations))
            )
            offset = arena.size
            arena.size += -(-descriptor.nbytes // _ALIGNMENT) * _ALIGNMENT
            arena.members += 1
            self._slots[descriptor.name] = _Slot(
                descriptor, arena, offset, arena.size
            )

    @contextmanager
    def stage(self) -> Iterator[GraphInputStaging]:
        """Opens one forward's staging scope and sends it on the way out.

        A body that raises sends nothing, since its host writes may be
        incomplete.
        """
        staging = GraphInputStaging(self._slots)
        yield staging
        staging.send()
