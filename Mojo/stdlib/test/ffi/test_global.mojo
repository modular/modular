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

from std.ffi import _Global
from std.memory import Pointer
from std.os import abort
from std.testing import TestSuite, assert_true


def _init_flag() -> Bool:
    return False


# Side-channel flag: _TrackableDel.__deinit__ sets this to True.
comptime _DEINIT_FLAG = _Global["_test_deinit_flag", _init_flag]


def _get_flag_ptr() -> Pointer[Bool, MutUntrackedOrigin]:
    try:
        return _DEINIT_FLAG.get_or_create_ptr()
    except:
        abort("failed to create _DEINIT_FLAG global")


struct _TrackableDel(Deinitable, Movable):
    var _value: Int

    def __init__(out self):
        self._value = 42

    def __deinit__(deinit self):
        _get_flag_ptr()[] = True


def _make_trackable() -> _TrackableDel:
    return _TrackableDel()


comptime _TRACKABLE_GLOBAL = _Global["_test_trackable_global", _make_trackable]


def test_deinit_wrapper_runs_destructor() raises:
    # Reset the flag.
    _get_flag_ptr()[] = False

    # Create the global (calls _init_wrapper under the hood).
    var opaque_ptr = _TRACKABLE_GLOBAL._init_wrapper()
    assert_true(Bool(opaque_ptr), msg="init_wrapper should return non-null")
    assert_true(not _get_flag_ptr()[], msg="destructor should not have run yet")

    # Tear down the global (calls _deinit_wrapper).
    _TRACKABLE_GLOBAL._deinit_wrapper(opaque_ptr)
    assert_true(
        _get_flag_ptr()[], msg="destructor should have run after deinit_wrapper"
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
