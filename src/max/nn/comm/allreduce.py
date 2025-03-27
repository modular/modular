# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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
"""Allreduce module definitions."""

from __future__ import annotations

import weakref
from collections.abc import Iterable

import numpy as np
from max.driver import Accelerator
from max.dtype import DType
from max.graph import (
    BufferValue,
    DeviceKind,
    DeviceRef,
    Graph,
    TensorValue,
    ops,
)
from max.nn.layer import Module


class Allreduce(Module):
    """Layer to perform allreduce operation with automatic implementation selection.

    Automatically chooses between peer-to-peer optimized allreduce and naive
    device-to-device transfer based on accelerator connectivity.

    Args:
        num_accelerators: Number of accelerators participating in the allreduce operation
    """

    devices: list[Accelerator]
    """List of accelerators involved in the allreduce operation."""

    signals: Signals
    """Signals for device peer-to-peer communication."""

    def __init__(
        self, num_accelerators: int, signals: Signals | None = None
    ) -> None:
        """Initialize the Allreduce layer with a specified number of accelerators.

        Args:
            num_accelerators: Number of accelerators to use for allreduce
            signals: Optional `Signals` class which defines the buffers used to
                communicate between devices. If your model contains multiple
                `Allreduce` modules, you should create a single `Signals` class
                and pass it to each `Allreduce`.

        Raises:
            ValueError: If num_accelerators is less than 1
        """
        super().__init__()
        if num_accelerators < 1:
            raise ValueError("At least one accelerator required for Allreduce")

        self.devices = [Accelerator(id=id) for id in range(num_accelerators)]
        if signals and not (len(signals.devices) == num_accelerators):
            raise ValueError(
                "Signals must contain the same number of devices "
                "as this allreduce."
            )
        self.signals = signals or Signals(
            DeviceRef.GPU(id) for id in range(num_accelerators)
        )

    def __call__(self, inputs: Iterable[TensorValue]) -> list[TensorValue]:
        """Performs allreduce operation with automatic implementation selection.

        Args:
            inputs: Distributed tensor values to reduce

        Returns:
            List of reduced tensors, one per device
        """
        return ops.allreduce.sum(inputs, self.signals.buffers())

    def _p2p_available(self) -> bool:
        """Check if peer-to-peer communication is available between devices."""
        # Implementation note: Currently checks first two devices as proxy
        # for full p2p connectivity. May need expansion for multi-device topologies.
        return self.devices[0].can_access(self.devices[1])


class Signals:
    """Signal buffers used for peer-to-peer communication in allreduce.

    Device code uses these buffers by enabling peer-to-peer access.
    Then thread blocks use the buffers to implement barriers for
    synchronization, and to hold intermediate communication results.
    """

    NUM_BYTES = (1 + 128) * 1024 * 1024
    """The size of the signal buffers used for communication in allreduce."""
    # NOTE: ``NUM_BYTES`` must stay in sync with the size of the ``Signal``
    # Mojo struct + the size of the intermediate buffer for communication.

    devices: list[DeviceRef]
    """List of graph devices that these signals communicate between."""

    def __init__(self, devices: Iterable[DeviceRef]) -> None:
        """
        Args:
            num_gpus: Number of GPUs involved in the allreduce.
        """
        # Convert the iterable to a list since we iterate over it twice.
        devices = list(devices)
        if not all(dev.device_type == DeviceKind.GPU for dev in devices):
            msg = "peer-to-peer signals should be constructed for accelerators"
            raise TypeError(msg)

        self.devices = devices

        # Cache the buffers created for each device and graph.
        self._cached_buffers: weakref.WeakKeyDictionary[
            weakref.ref[Graph], dict[DeviceRef, BufferValue]
        ] = weakref.WeakKeyDictionary()

    def buffers(self) -> list[BufferValue]:
        """Allocates and returns buffers used for communication in allreduce."""
        # Find signals buffer from the cache or create and initialize new
        # buffers.
        graph = Graph.current
        device_signal_buffers = self._cached_buffers.setdefault(graph, {})
        signal_buffers = []
        for device in self.devices:
            if cached_buffer := device_signal_buffers.get(device):
                signal_buffers.append(cached_buffer)
            else:
                buffer = ops.buffer_create(
                    shape=(Signals.NUM_BYTES,),
                    dtype=DType.uint8,
                    device=device,
                )
                ops.buffer_store(
                    buffer,
                    # TODO(GEX-1988): Use Tensor instead of numpy here.
                    ops.constant(
                        np.zeros((Signals.NUM_BYTES,), dtype=np.uint8),
                        DType.uint8,
                    ).to(device),
                )
                signal_buffers.append(buffer)
                device_signal_buffers[device] = buffer

        return signal_buffers
