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
"""Tests GPU range annotations when the tracing library is not on its paths.

Serving images built on the CUDA base image ship no `libnvToolsExt`. Built with
detailed profiling, the first range used to abort the process with
"symbol not found: nvtxRangeStartEx". Ranges must be no-ops then, but still
record when another library has already loaded the range API.
"""

from std.ffi import OwnedDLHandle, _Global
from std.pathlib import Path
from std.testing import assert_equal, assert_false, assert_true, TestSuite

from max.gpu.host import (
    LIBRARY_PATHS,
    _end_range,
    _load_library,
    _mark,
    _start_range,
)


def _missing_library() -> OwnedDLHandle:
    return OwnedDLHandle(unsafe_uninitialized=True)


def test_ranges_without_library() raises:
    # Globals are created once per name, so claiming the tracing library's
    # global first makes it hold the null handle that `_init_dylib` stores
    # when no library loads, even on a machine that has one.
    _ = _Global["GPU_TRACING_LIBRARY", _missing_library].get_or_create_ptr()

    var id = _start_range(message="kernel/(shape)")
    assert_equal(id, 0)
    _mark(message="mark")
    _end_range(id)


def test_load_library_falls_back_to_process() raises:
    var missing: List[Path] = ["/nonexistent/libnvToolsExt.so"]
    assert_false(Bool(_load_library(missing)))

    # Loading from the real paths puts the range API in the process's global
    # scope, as when another library of the process has loaded it.
    var real = _load_library(materialize[LIBRARY_PATHS]())
    if not real:
        print("skipped the fallback check: no tracing library on this machine")
        return
    assert_true(Bool(_load_library(missing)))
    _ = real^


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
