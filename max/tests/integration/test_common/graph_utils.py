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

from max.driver import Buffer, accelerator_api, accelerator_architecture_name
from max.graph import BufferValue, TensorValue, Value

# Asked of the driver rather than of NVML, so a test that only needs to know
# which arch it is compiling for can answer it on a host with no GPU attached:
# under virtual devices the driver reports the arch it was configured with,
# where NVML sees no devices at all.


def is_h100_h200() -> bool:
    """Checks if this is an H100 or H200 GPU."""
    return accelerator_architecture_name().startswith("sm_90")


def is_b100_b200() -> bool:
    """Checks if this is an B100 or B200 GPU."""
    return accelerator_architecture_name().startswith("sm_100")


def is_nvidia_gpu() -> bool:
    """Checks if the GPU is an NVIDIA GPU."""
    return accelerator_api() == "cuda"


def gpu_warp_size() -> int:
    """Returns the warp/wavefront size for the current GPU."""
    return 32 if is_nvidia_gpu() else 64


def is_a10() -> bool:
    """Checks if this is an A10 GPU.

    `sm_86` covers several consumer and datacenter parts; the A10 is the only
    one of them in our fleet.
    """
    return accelerator_architecture_name() == "sm_86"


def are_all_tensors_iterable(
    it: Iterable[Buffer],
) -> TypeGuard[Iterable[Buffer]]:
    return all(isinstance(value, Buffer) for value in it)


def are_all_tensors_sequence(
    it: Sequence[Buffer],
) -> TypeGuard[Sequence[Buffer]]:
    return all(isinstance(value, Buffer) for value in it)


def are_all_buffer_values_sequence(
    it: Sequence[Value],  # type: ignore[type-arg]
) -> TypeGuard[Sequence[BufferValue]]:
    return all(isinstance(value, BufferValue) for value in it)


def are_all_tensor_values_iterable(
    it: Iterable[Value],  # type: ignore[type-arg]
) -> TypeGuard[Iterable[TensorValue]]:
    return all(isinstance(value, TensorValue) for value in it)
