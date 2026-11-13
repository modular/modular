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
"""Selects the `DeviceContext` backend behind `max.gpu.host`.

The build system picks the backend by replacing this file with
`max/_driver/max_license/device_context_hal/_dcc_overlay.mojo` under
--config=hal-device-context. What differs between the two builds is the
constant below and the `DeviceGraphInput` trait, which names a different trait
per backend and so cannot be selected by a conditional alias.
"""

comptime HAL_DEVICE_CONTEXT = False
"""Whether `max.gpu.host` is the HAL `DeviceContext`."""

from ._legacy.device_graph import LegacyDeviceGraphInput as DeviceGraphInput
