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

from collections.abc import Iterable

from max.driver import Accelerator, Tensor
from max.dtype import DType
from max.graph import BufferType, DeviceKind, DeviceRef


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

    def buffers(self) -> list[Tensor]:
        """Allocates and returns buffers used for communication in allreduce."""
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
