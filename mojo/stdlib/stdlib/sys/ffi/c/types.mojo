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
"""C POSIX types."""

from sys.info import is_32bit, is_64bit, os_is_windows
from sys.ffi import external_call
from os import abort
from utils import StaticTuple
from memory import memcpy, UnsafePointer

# ===----------------------------------------------------------------------=== #
# Base Types
# ===----------------------------------------------------------------------=== #


struct C:
    """C types. This assumes that the platform is 32 or 64 bit, and char is
    always 8 bit (POSIX standard).
    """

    alias char = Int8
    """Type: `char`. The signedness of `char` is platform specific. Most
    systems, including x86 GNU/Linux and Windows, use `signed char`, but those
    based on PowerPC and ARM processors typically use `unsigned char`."""
    alias s_char = Int8
    """Type: `signed char`."""
    alias u_char = UInt8
    """Type: `unsigned char`."""
    alias short = Int16
    """Type: `short`."""
    alias u_short = UInt16
    """Type: `unsigned short`."""
    alias int = Int32
    """Type: `int`."""
    alias u_int = UInt32
    """Type: `unsigned int`."""
    alias long = Scalar[_c_long_dtype()]
    """Type: `long`."""
    alias u_long = Scalar[_c_u_long_dtype()]
    """Type: `unsigned long`."""
    alias long_long = Int64
    """Type: `long long`."""
    alias u_long_long = UInt64
    """Type: `unsigned long long`."""
    alias float = Float32
    """Type: `float`."""
    alias double = Float64
    """Type: `double`."""
    alias void = Int8
    """Type: `void`."""
    alias NULL = UnsafePointer[Self.void]()
    """Constant: NULL pointer."""
    alias ptr_addr = Int
    """Type: A Pointer Address."""


alias size_t = UInt
"""Type: `size_t`."""
alias ssize_t = Int
"""Type: `ssize_t`."""


# ===----------------------------------------------------------------------=== #
# Utils
# ===----------------------------------------------------------------------=== #


fn _c_long_dtype() -> DType:
    # https://en.wikipedia.org/wiki/64-bit_computing#64-bit_data_models

    @parameter
    if is_64bit() and os_is_windows():
        return DType.int32  # LLP64
    elif is_64bit():
        return DType.int64  # LP64
    elif is_32bit():
        return DType.int32  # ILP32
    else:
        constrained[False, "size of C `long` is unknown on this target"]()
        return abort[DType]()


fn _c_u_long_dtype() -> DType:
    # https://en.wikipedia.org/wiki/64-bit_computing#64-bit_data_models

    @parameter
    if is_64bit() and os_is_windows():
        return DType.uint32  # LLP64
    elif is_64bit():
        return DType.uint64  # LP64
    elif is_32bit():
        return DType.uint32  # ILP32
    else:
        constrained[False, "size of C `long long` is unknown on this target"]()
        return abort[DType]()
