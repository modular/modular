# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the MAX License:
# https://modular.com/legal/max-license
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Selects the HAL `DeviceContext` backend behind `max.gpu.host`.

Replaces `max/gpu/host/_overlay.mojo` under --config=hal-device-context (see
`:dcc_overlay` in this package's BUILD.bazel) and holds the same two names: the
backend constant and the `DeviceGraphInput` trait.
"""

comptime HAL_DEVICE_CONTEXT = True
"""Whether `max.gpu.host` is the HAL `DeviceContext`."""

from max._driver.max_license.device_context_hal.device_graph_hal import (
    HALDeviceGraphInput as DeviceGraphInput,
)
