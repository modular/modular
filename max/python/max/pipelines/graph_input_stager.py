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
from dataclasses import dataclass

from max.driver import (
    Buffer,
    Device,
    DevicePinnedBuffer,
    copy_pinned_to_destinations,
)
from max.dtype import DType


@dataclass(frozen=True)
class InputDescriptor:
    """One graph input: its name, the widest it gets, and where it goes."""

    name: str
    """Prefix it with the replica index for anything a replica owns."""

    dtype: DType

    max_shape: tuple[int, ...]
    """The shape at the pipeline's batching limits. A step that outgrows it
    raises."""

    destinations: Sequence[Device]
    """Where a step's value is copied, in the order
    :meth:`GraphInputStaging.get` returns them. Host staging is allocated on
    the first."""


class _Declaration:
    """One described input, and the device allocation it has acquired."""

    def __init__(self, descriptor: InputDescriptor) -> None:
        self.descriptor = descriptor
        self.capacity = math.prod(descriptor.max_shape)
        self.backings: tuple[Buffer, ...] | None = None

    def views(self, shape: tuple[int, ...]) -> tuple[Buffer, ...]:
        """This shape over the stable allocation, one view per destination.

        Raises:
            RuntimeError: If ``shape`` outgrows the described maximum.
        """
        name = self.descriptor.name
        num_elements = math.prod(shape)
        if num_elements > self.capacity:
            raise RuntimeError(
                f"Graph input {name!r} needs {num_elements} elements "
                f"(shape={list(shape)}), beyond the {self.capacity} it was "
                f"sized for (max_shape={list(self.descriptor.max_shape)}). "
                f"The batching dimensions this input was described from do "
                f"not bound it."
            )
        if self.backings is None:
            self.backings = tuple(
                Buffer(
                    shape=(self.capacity,),
                    dtype=self.descriptor.dtype,
                    device=device,
                )
                for device in self.descriptor.destinations
            )
        return tuple(
            backing[:num_elements].view(self.descriptor.dtype, shape)
            for backing in self.backings
        )


class GraphInputStaging:
    """One forward's staging, handed out by :meth:`GraphInputStager.stage`.

    Everything asked for through this object is sent when the scope closes,
    and nothing else is: an input this forward did not ask for keeps whatever
    the graph last read. There is no way to reach :meth:`get` without a scope,
    so an input cannot be filled without something sending it.
    """

    def __init__(self, declared: Mapping[str, _Declaration]) -> None:
        self._declared = declared
        self._staged: dict[str, tuple[Buffer, tuple[Buffer, ...]]] = {}

    def get(
        self, name: str, shape: tuple[int, ...]
    ) -> tuple[Buffer, tuple[Buffer, ...]]:
        """Returns this step's host staging for ``name`` and its destinations.

        The host buffer is fresh every step and never reused: an H2D copies
        what the buffer holds when the copy runs, so the next step's host
        writes could overtake this one's.

        Raises:
            KeyError: If ``name`` was never described.
            RuntimeError: If ``shape`` outgrows the described maximum, or
                disagrees with an earlier call in this scope.
        """
        shape = tuple(shape)
        staged = self._staged.get(name)
        if staged is not None:
            # Every shard of a replica asks for the metadata they share, so
            # a second call is the same input: one host write, one fan-out.
            if tuple(staged[0].shape) != shape:
                raise RuntimeError(
                    f"Graph input {name!r} was staged as "
                    f"{list(staged[0].shape)} and again as {list(shape)} in "
                    "one forward"
                )
            return staged

        declaration = self._declared[name]
        destinations = declaration.views(shape)
        # Pinned staging makes the H2D async; a host device cannot pin, and
        # has no transfer to make asynchronous either.
        staging_device = declaration.descriptor.destinations[0]
        buffer_cls = Buffer if staging_device.is_host else DevicePinnedBuffer
        host = buffer_cls(
            shape=shape,
            dtype=declaration.descriptor.dtype,
            device=staging_device,
        )
        self._staged[name] = (host, destinations)
        return host, destinations

    def send(self) -> None:
        """Copies what this scope staged to its devices.

        Called by :meth:`GraphInputStager.stage` on the way out.
        :func:`~max.driver.copy_pinned_to_destinations` makes the staging
        device wait for the other shards, so staging is not recycled while
        their copies are still reading it.
        """
        for host, destinations in self._staged.values():
            copy_pinned_to_destinations(host, destinations)


class GraphInputStager:
    """Device input buffers sized to the largest batch the pipeline allows.

    A step opens a :meth:`stage` scope, asks for the prefix it needs, fills
    the host half, and every transfer it staged is issued when the scope
    closes.

    One instance serves every KV leaf, tensor-parallel shard and data-parallel
    replica. Leaf names are distinct and shards are the destinations of one
    name, but replicas need a name prefix: ``Device.__eq__`` compares label and
    id, a CPU device's id is always 0, and the device list is split per replica
    rather than being disjoint, so a data-parallel pipeline on CPU hands every
    replica the same ``Device``.

    The device side of an input is allocated lazily and reused for the life of
    the process; its allocation is what a captured graph binds. Host staging is
    allocated fresh every step and released with the scope, or the overlap
    scheduler's next step could write into a copy still in flight.

    Whether the staging is pinned is this class's business: a host device
    cannot pin, and callers write it the same way either way.

    Args:
        inputs: Every input a forward can stage. A name appears once.

    Raises:
        ValueError: If a name repeats, or an input has no destination.
    """

    def __init__(self, inputs: Iterable[InputDescriptor]) -> None:
        self._declared: dict[str, _Declaration] = {}
        for descriptor in inputs:
            if not descriptor.destinations:
                raise ValueError(
                    f"Graph input {descriptor.name!r} needs at least one "
                    "destination device"
                )
            if descriptor.name in self._declared:
                raise ValueError(
                    f"Graph input {descriptor.name!r} is described twice"
                )
            self._declared[descriptor.name] = _Declaration(descriptor)

    @contextmanager
    def stage(self) -> Iterator[GraphInputStaging]:
        """Scopes one forward's staging, and sends it on the way out.

        Leaving the scope issues every transfer the forward staged at once,
        so it pays the fan-out once however many inputs it wrote. A body that
        raises sends nothing, since its host writes did not finish.
        """
        staging = GraphInputStaging(self._declared)
        yield staging
        staging.send()
