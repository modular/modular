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
"""The NeoDriver client API.

Re-exports the HAL client API from `max_license.hal`. Its submodules, such as
`status`, `extensions`, and `execution_config`, are reached through
`max._driver.max_license.hal`.
"""

from .max_license.hal import (
    Buffer,
    BufferView,
    Context,
    Device,
    Driver,
    Event,
    FunctionHandle,
    HALError,
    Queue,
    RawDriver,
    RuntimeBundle,
    STATUS_SUCCESS,
    Stream,
    copy,
    fill,
    get_device_spec,
    set_memory,
)
