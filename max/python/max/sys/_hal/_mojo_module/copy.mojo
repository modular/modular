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
"""Python binding for the module-level HAL `copy`."""

from std.python import PythonObject
from _hal.copy import copy as _hal_copy

from .buffer import Buffer


def copy(dst_obj: PythonObject, src_obj: PythonObject) raises:
    """Synchronously copies device buffer `src` into `dst`.

    Projects the module-level Mojo HAL `copy(dst, src)` — residency dispatch and
    the blocking, stream-less plugin copy ops all happen on the Mojo side.
    """
    var dst_ptr = dst_obj.downcast_value_ptr[Buffer]()
    var src_ptr = src_obj.downcast_value_ptr[Buffer]()
    _hal_copy(dst=dst_ptr[]._hal, src=src_ptr[]._hal)
