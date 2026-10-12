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

"""Device programming for MAX: allocate buffers, manage devices, and
dispatch work through device queues.
"""

import contextlib
from collections.abc import Iterator

from max._core import __version__
from max._core.driver import (
    CompletionFlag,
    DeviceEvent,
    HostHazardError,
    LaunchTraceEntry,
    SymmetricPool,
    Usage,
    __unsafe_pack_py_host_func,
    begin_launch_trace,
    enable_all_peer_access,
    get_virtual_cpu_target,
    get_virtual_device_api,
    get_virtual_device_count,
    get_virtual_device_target_arch,
    is_virtual_device_mode,
    set_virtual_cpu_target,
    set_virtual_device_api,
    set_virtual_device_count,
    set_virtual_device_target_arch,
    take_launch_trace,
)
from max._core_types.driver import DLPackArray

from .buffer import (
    Buffer,
    _ChunkedStagingRegion,
    batch_inplace_copy,
    copy_pinned_to_destinations,
    load_max_buffer,
)
from .driver import (
    CPU,
    Accelerator,
    Device,
    DeviceQueue,
    DeviceSpec,
    accelerator_api,
    accelerator_architecture_name,
    accelerator_count,
    calculate_virtual_device_count,
    calculate_virtual_device_count_from_cli,
    devices_exist,
    load_devices,
    scan_available_devices,
)

# nb::exception types cannot carry a docstring from C++, so document the
# exception at its re-export here.
HostHazardError.__doc__ = """Raised when host code waits on a completion whose producer failed.

Host-hazard tracking orders host access behind the device work that writes a
buffer. When that work fails, the completion is poisoned, and waiting on it
(or touching the buffers it covers) raises this error without blocking.
The message carries the failed producer's rendered traceback.
"""


@contextlib.contextmanager
def launch_trace() -> Iterator[list[LaunchTraceEntry]]:
    """Records the device operations enqueued within the ``with`` block.

    Wraps :func:`begin_launch_trace` and :func:`take_launch_trace` so the
    process-global recording is always stopped, even if the block raises.
    Only CUDA and HIP devices record entries; on other devices the list
    stays empty.

    The yielded list is empty while the block runs and is filled with the
    recorded :class:`LaunchTraceEntry` values once the block exits, in
    enqueue order across all streams.

    For example, capture the device operations a block of code enqueues on
    an accelerator:

    .. code-block:: python

        import numpy as np

        from max import driver
        from max.dtype import DType

        device = (
            driver.Accelerator() if driver.accelerator_count() else driver.CPU()
        )
        src = driver.Buffer.from_numpy(np.zeros(4, dtype=np.float32))
        dst = driver.Buffer(dtype=DType.float32, shape=[4], device=device)

        with driver.launch_trace() as entries:
            dst.inplace_copy_from(src)

        print(len(entries))

    Yields:
        The operations enqueued within the block. Empty until the block
        exits.
    """
    entries: list[LaunchTraceEntry] = []
    begin_launch_trace()
    try:
        yield entries
    finally:
        entries.extend(take_launch_trace())


__all__ = [
    "CPU",
    "Accelerator",
    "Buffer",
    "CompletionFlag",
    "DLPackArray",
    "Device",
    "DeviceEvent",
    "DeviceQueue",
    "DeviceSpec",
    "HostHazardError",
    "LaunchTraceEntry",
    "SymmetricPool",
    "Usage",
    "accelerator_api",
    "accelerator_architecture_name",
    "accelerator_count",
    "batch_inplace_copy",
    "begin_launch_trace",
    "calculate_virtual_device_count",
    "calculate_virtual_device_count_from_cli",
    "copy_pinned_to_destinations",
    "devices_exist",
    "enable_all_peer_access",
    "get_virtual_cpu_target",
    "get_virtual_device_api",
    "get_virtual_device_count",
    "get_virtual_device_target_arch",
    "is_virtual_device_mode",
    "launch_trace",
    "load_devices",
    "load_max_buffer",
    "scan_available_devices",
    "set_virtual_cpu_target",
    "set_virtual_device_api",
    "set_virtual_device_count",
    "set_virtual_device_target_arch",
    "take_launch_trace",
]
