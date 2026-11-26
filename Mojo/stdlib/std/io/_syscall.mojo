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
"""Turns failed C calls into `IOError`s."""

from std.ffi import c_int
from std.sys import CompilationTarget

from .error import IOError, IOErrorKind


@inline(.always)
def cap_io_length(length: Int) -> Int:
    """Returns `length`, capped at the most bytes one `read`, `write`, `recv`
    or `send` call accepts.

    macOS fails a call of more than `c_int.MAX` bytes with `EINVAL` instead of
    transferring fewer.
    """
    comptime if CompilationTarget.is_macos():
        return min(length, Int(c_int.MAX))
    else:
        return length


@inline(.always)
def check[
    dtype: DType
](result: Scalar[dtype], operation: StaticString) raises IOError -> type_of(
    result
):
    """Returns `result`, or raises the current `errno` if it's -1."""
    comptime assert dtype.is_integral() and dtype.is_signed()
    if result == -1:
        raise IOError.last_os_error(operation)
    return result


@inline(.always)
def retry_io[
    R: AnyType,
    //,
    F: def() raises IOError -> R,
    *,
    retry_kinds: Array[IOErrorKind, _],
](mut f: F) raises IOError -> R:
    """Calls `f` until it doesn't raise an error of a kind in `retry_kinds`,
    and returns its result.

    An error of any other kind is raised.
    """
    while True:
        try:
            return f()
        except e:
            var kind = e.kind()
            var retry = False
            comptime for retry_kind in retry_kinds:
                retry = retry or kind == retry_kind
            if not retry:
                raise e


comptime retry_io_interrupted = retry_io[retry_kinds=[.interrupted], ...]
"""Calls `f` until it doesn't raise an `interrupted` error, and returns its
result.
"""
