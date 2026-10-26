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
"""Selects the HAL `DeviceContext` behind `max.gpu.host`.

Replaces `max/gpu/host/_overlay.mojo` under --config=hal-device-context (see
`:dcc_overlay` in this package's BUILD.bazel), and exports the same names as
that file, plus the HAL-only ones. It imports absolutely so it resolves the same
staged there as here, where it also compiles in every configuration.
"""

from max._driver.max_license.device_context_hal.device_context_hal import *
from max._driver.max_license.device_context_hal.device_context_hal import (
    _CString,
    _DeviceBufferCpp,
    _DeviceContextPtr,
    _DeviceContextScopeHAL,
    _DeviceFunctionPtr,
    _DumpPath,
    _check_dim,
    _checked,
)
from max._driver.max_license.device_context_hal.device_graph_hal import *
from max._driver.max_license.device_context_hal._nvidia_cuda_hal import *
from max._driver.max_license.device_context_hal._nvidia_cuda_hal import (
    _CUstream_st,
)
