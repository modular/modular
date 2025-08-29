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
"""Op implementation for transfer_to."""

from __future__ import annotations

from max.mlir.dialects import rmo

from ..graph import Graph
from ..type import Device, DeviceRef, TensorType
from ..value import StrongTensorValueLike, TensorValue


def transfer_to(
    x: StrongTensorValueLike, device: Device | DeviceRef
) -> TensorValue:
    """Device-to-Device transfer operation.

    This op transfers the input tensor from its current device over to another. A device represents a
    computation unit, like CPU, GPU, etc. This op is useful for instance when working with
    accelerators, like GPU, where for instance one may need to move data from GPU to GPU, or
    from one GPU to CPU.

    Args:
        x: The input tensor to transfer.
        device: The device to transfer to.

    Returns:
        A tensor transferred to device specified.
    """
    x = TensorValue(x)
    device = DeviceRef.from_device(device)

    if device == x.type.device:
        return x

    return Graph.current._add_op(
        rmo.mo_transfer,
        TensorType(dtype=x.dtype, shape=x.shape, device=device).to_mlir(),
        x,
    )[0].tensor
