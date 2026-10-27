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
"""Selects the `DeviceContext` implementation behind `max.gpu.host`.

`max.gpu.host` takes the `device_context`, `device_graph`, and
`_nvidia_cuda` names from here. This variant names the legacy
implementation; an overlay replaces this file to swap in another one that
exports the same names.
"""

from ._legacy.device_context import *
from ._legacy.device_context import (
    _CString,
    _DeviceBufferCpp,
    _DeviceContextPtr,
    _DeviceFunctionPtr,
    _DumpPath,
    _checked,
)
from ._legacy.device_graph import *
from ._legacy._nvidia_cuda import *
from ._legacy._nvidia_cuda import _CUstream_st
