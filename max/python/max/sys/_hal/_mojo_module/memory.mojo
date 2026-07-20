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
"""Python bindings for the module-level HAL `copy` / `set_memory` / `fill`."""

from std.python import PythonObject
from _hal.memory import (
    copy as _hal_copy,
    fill as _hal_fill,
    set_memory as _hal_set_memory,
)

from .buffer import Buffer


def copy(dst_obj: PythonObject, src_obj: PythonObject) raises:
    """Synchronously copies device buffer `src` into `dst`.

    Projects the module-level Mojo HAL `copy(dst, src)` — residency dispatch and
    the blocking, stream-less plugin copy ops all happen on the Mojo side.
    """
    var dst_ptr = dst_obj.downcast_value_ptr[Buffer]()
    var src_ptr = src_obj.downcast_value_ptr[Buffer]()
    _hal_copy(dst=dst_ptr[]._hal, src=src_ptr[]._hal)


def set_memory(dst_obj: PythonObject, value_obj: PythonObject) raises:
    """Synchronously sets every byte of device buffer `dst` to `value`.

    Projects the module-level Mojo HAL `set_memory(dst, value)` — the blocking,
    stream-less plugin op happens on the Mojo side; no queue is created.
    """
    var dst_ptr = dst_obj.downcast_value_ptr[Buffer]()
    _hal_set_memory(dst=dst_ptr[]._hal, value=UInt8(Int(py=value_obj)))


def fill(
    dst_obj: PythonObject,
    value_obj: PythonObject,
    value_size_obj: PythonObject,
) raises:
    """Synchronously fills device buffer `dst` with a repeated value.

    Projects the module-level Mojo HAL `fill(dst, value, value_size)` — the
    blocking, stream-less plugin op happens on the Mojo side; no queue is
    created.
    """
    var dst_ptr = dst_obj.downcast_value_ptr[Buffer]()
    _hal_fill(
        dst=dst_ptr[]._hal,
        value=UInt64(Int(py=value_obj)),
        value_size=UInt64(Int(py=value_size_obj)),
    )
