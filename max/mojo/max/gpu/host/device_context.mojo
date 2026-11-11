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
"""This module provides functionality for interacting with accelerators. In
particular the
[`DeviceContext`](/api/mojo/max/gpu/host/device_context/DeviceContext/) struct,
which represents a single stream of execution on a given accelerator. You can
use this struct to allocate accelerator memory, copy data to and from the
accelerator, and compile and execute functions on the accelerator."""


@__doc_inline
from max._driver.max_license.device_context_dispatch.device_context import (
    DeviceContext,
)
from max._driver.max_license.device_context_dispatch.device_buffer import (
    DeviceBuffer,
)
from max._driver.max_license.device_context_dispatch.host_buffer import (
    HostBuffer,
)
from max._driver.max_license.device_context_dispatch.device_pointer import (
    DevicePointer,
)
from max._driver.max_license.device_context_dispatch.function import (
    DeviceExternalFunction,
    DeviceFunction,
)
from max._driver.max_license.device_context_dispatch.host_types import (
    CompletionFlag,
    Device,
    DeviceContextArray,
    DeviceEvent,
    DeviceMulticastBuffer,
    DeviceStream,
)
from ._overlay import (
    DefaultDeviceTypeEncoder,
)
from max.gpu.host._driver_common import StreamPriorityRange
