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

# RUN: %mojo %s

# A function value with a bound `StaticString` parameter keeps that string when
# it is read back out of a comptime `Optional`. The string reaches the
# interpreter as a symbol pointing into the value's memory space, and its blob
# has to be found by its index in the interpreter table it was loaded into, not
# by its slot in the memory space. The two differ whenever that table already
# held blobs or the space spans several tables. Regression test for MOCO-5289.

from std.collections.string.string_span import get_static_string
from std.sys import llvm_intrinsic
from std.testing import assert_equal

comptime HookFn = def[dtype: DType, width: SIMDLength, //](
    SIMD[dtype, width]
) thin -> SIMD[dtype, width]


def doubled_if_named[
    dtype: DType, width: SIMDLength, //, name: StaticString
](x: SIMD[dtype, width]) -> SIMD[dtype, width]:
    """Reads the bound string at run time."""
    if String(name) == "sqrt":
        return x * 2
    return x


def lowered[
    dtype: DType, width: SIMDLength, //, name: StaticString
](x: SIMD[dtype, width]) -> SIMD[dtype, width]:
    """Reads the bound string at compile time."""
    return llvm_intrinsic[
        get_static_string["llvm.", name](), type_of(x), has_side_effect=False
    ](x)


comptime run_time_hook = Optional[HookFn](doubled_if_named[name="sqrt"])
comptime compile_time_hook = Optional[HookFn](lowered[name="sqrt"])


def main() raises:
    comptime if case .Some(f) = run_time_hook:
        assert_equal(f(Float32(1)), 2)

    comptime if case .Some(f) = compile_time_hook:
        assert_equal(f(Float32(16)), 4)

    # `value()` returns the payload by reference, a separate read-back path.
    comptime g = run_time_hook.value()
    assert_equal(g(Float32(1)), 2)

    comptime h = compile_time_hook.value()
    assert_equal(h(Float32(16)), 4)
