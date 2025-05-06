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

from collections.abc import Iterable

from max.driver import Accelerator, Tensor
from max.dtype import DType
from max.graph import (
    BufferType,
    BufferValue,
    DeviceKind,
    DeviceRef,
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

    def __init__(self, num_accelerators: int) -> None:
        """Initialize the Allreduce layer with a specified number of accelerators.

        Args:
            num_accelerators: Number of accelerators to use for allreduce

        Raises:
            ValueError: If num_accelerators is less than 1
        """
        super().__init__()
        if num_accelerators < 1:
            raise ValueError("At least one accelerator required for Allreduce")

        self.devices = [Accelerator(id=id) for id in range(num_accelerators)]

    def __call__(
        self, inputs: Iterable[TensorValue], signal_buffers: list[BufferValue]
    ) -> list[TensorValue]:
        """Performs allreduce operation with automatic implementation selection.

        Args:
            inputs: Distributed tensor values to reduce
            signal_buffers: Buffers for peer-to-peer communication when using
                optimized allreduce.

        Returns:
            List of reduced tensors, one per device
        """
        return ops.allreduce.sum(inputs, signal_buffers)

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

    NUM_BYTES = (1 + 512) * 1024 * 1024
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

    def buffers(self) -> list[Tensor]:
        """Allocates and returns buffers used for communication in allreduce."""
        # Contents of signal buffer should be filled with zeros.
        return [
            Tensor.zeros(
                shape=(Signals.NUM_BYTES,),
                dtype=DType.uint8,
                device=Accelerator(id=dev.id),
            )
            for dev in self.devices
        ]

    def input_types(self) -> list[BufferType]:
        """Gets graph input types corresponding to these signal buffers."""
        return [
            BufferType(
                dtype=DType.uint8, shape=(Signals.NUM_BYTES,), device=dev
            )
            for dev in self.devices
        ]
