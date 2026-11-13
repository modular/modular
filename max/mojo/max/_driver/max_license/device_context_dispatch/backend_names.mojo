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
"""The backend-specific names that `max.gpu.host` exports, selected by backend.

The wrappers cover the types with methods, and the handle types and helpers
that both backends share live in `max.gpu.host._driver_handles`,
`_vendor_handles` and `_driver_common`. What is left are the two names that each
backend defines for itself. Each name here is the linked backend's, so code that
imports it from `max.gpu.host` never names a backend.
"""

from max.gpu.host._vendor_handles import CUcontext

from max._driver.max_license.device_context_hal._nvidia_cuda_hal import (
    CUDA_get_current_context as _hal_CUDA_get_current_context,
)
from max._driver.max_license.device_context_hal.device_context_hal import (
    DefaultDeviceTypeEncoder as _HALDefaultDeviceTypeEncoder,
)
from max.gpu.host._legacy._nvidia_cuda import (
    CUDA_get_current_context as _legacy_CUDA_get_current_context,
)
from max.gpu.host._legacy.device_context import (
    DefaultDeviceTypeEncoder as _LegacyDefaultDeviceTypeEncoder,
)
from max.gpu.host._overlay import HAL_DEVICE_CONTEXT

comptime DefaultDeviceTypeEncoder = (
    _LegacyDefaultDeviceTypeEncoder if not HAL_DEVICE_CONTEXT else _HALDefaultDeviceTypeEncoder
)
"""The encoder kernels use for device arguments."""


def CUDA_get_current_context() raises -> CUcontext:
    """Returns the CUDA driver context that is current on this thread.

    Returns:
        The native `CUcontext`.

    Raises:
        If there is no current context, or the backend cannot report it.
    """
    comptime if not HAL_DEVICE_CONTEXT:
        return _legacy_CUDA_get_current_context()
    else:
        return _hal_CUDA_get_current_context()
