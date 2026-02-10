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

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TypeGuard

from max.driver import Buffer
from max.graph import BufferValue, TensorValue, Value
from nvitop import Device as NVITOPDevice


def is_h100_h200() -> bool:
    """Checks if this is an H100 or H200 GPU."""
    devices = NVITOPDevice.all()
    return bool(devices) and (
        "H100" in devices[0].name() or "H200" in devices[0].name()
    )


def is_b100_b200() -> bool:
    """Checks if this is an B100 or B200 GPU."""
    devices = NVITOPDevice.all()
    return bool(devices) and (
        "B100" in devices[0].name() or "B200" in devices[0].name()
    )


def is_nvidia_gpu() -> bool:
    """Checks if the GPU is an NVIDIA GPU."""
    return bool(NVITOPDevice.all())


def gpu_warp_size() -> int:
    """Returns the warp/wavefront size for the current GPU."""
    return 32 if is_nvidia_gpu() else 64


def is_a10() -> bool:
    """Checks if this is an A10 GPU."""
    devices = NVITOPDevice.all()
    return bool(devices) and "A10" in devices[0].name()


def are_all_tensors_iterable(
    it: Iterable[Buffer],
) -> TypeGuard[Iterable[Buffer]]:
    return all(isinstance(value, Buffer) for value in it)


def are_all_tensors_sequence(
    it: Sequence[Buffer],
) -> TypeGuard[Sequence[Buffer]]:
    return all(isinstance(value, Buffer) for value in it)


def are_all_buffer_values_sequence(
    it: Sequence[Value],
) -> TypeGuard[Sequence[BufferValue]]:
    return all(isinstance(value, BufferValue) for value in it)


def are_all_tensor_values_iterable(
    it: Iterable[Value],
) -> TypeGuard[Iterable[TensorValue]]:
    return all(isinstance(value, TensorValue) for value in it)


def are_all_tensor_values_sequence(
    it: Sequence[Value],
) -> TypeGuard[Sequence[TensorValue]]:
    return all(isinstance(value, TensorValue) for value in it)
