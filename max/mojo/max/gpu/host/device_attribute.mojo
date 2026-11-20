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
"""This module defines GPU device attributes that can be queried from CUDA-compatible devices.

The module provides the `DeviceAttribute` struct which encapsulates the various device
properties and capabilities that can be queried through the CUDA driver API. Each attribute
is represented as a constant with a corresponding integer value that maps to the CUDA
driver's attribute enumeration.

These attributes allow applications to query specific hardware capabilities and limitations
of GPU devices, such as maximum thread counts, memory sizes, compute capabilities, and
supported features.

:::note
See the [`DeviceContext`](/api/mojo/max/gpu/host/device_context/DeviceContext/) page
for examples that retrieve `DeviceAttribute` values.
:::
"""


@__doc_inline
from ._legacy.device_attribute import (
    DeviceAttribute,
)
