# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

from sys.ffi import c_int, c_long, c_long_long, c_ulong, c_ulong_long
from sys.info import CompilationTarget, is_64bit

from testing import assert_equal, assert_true

#
# Reference:
#     https://en.wikipedia.org/wiki/64-bit_computing#64-bit_data_models
#


def test_c_int_type():
    if is_64bit() and (
        CompilationTarget.is_macos()
        or CompilationTarget.is_linux()
        or CompilationTarget.is_windows()
    ):
        # `int` is always 32 bits on the modern 64-bit OSes.
        assert_equal(c_int.dtype, DType.int32)
    else:
        assert_true(False, "platform c_int size is untested")


def test_c_long_types():
    if is_64bit() and (
        CompilationTarget.is_macos() or CompilationTarget.is_linux()
    ):
        # `long` is 64 bits on macOS and Linux.
        assert_equal(c_long.dtype, DType.int64)
        assert_equal(c_ulong.dtype, DType.uint64)
    elif is_64bit() and CompilationTarget.is_windows():
        # `long` is 32 bits only on Windows.
        assert_equal(c_long.dtype, DType.int32)
        assert_equal(c_ulong.dtype, DType.uint32)
    else:
        assert_true(False, "platform c_long and c_ulong size is untested")


def test_c_long_long_types():
    if is_64bit() and (
        CompilationTarget.is_macos()
        or CompilationTarget.is_linux()
        or CompilationTarget.is_windows()
    ):
        assert_equal(c_long_long.dtype, DType.int64)
        assert_equal(c_ulong_long.dtype, DType.uint64)
    else:
        assert_true(
            False, "platform c_long_long and c_ulong_long size is untested"
        )


def main():
    test_c_int_type()
    test_c_long_types()
    test_c_long_long_types()
