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
"""Implements compile-time constraints.

These are Mojo built-ins, so you don't need to import them.
"""
from collections.string.string_slice import _get_kgen_string


@always_inline("nodebug")
fn constrained[cond: Bool, msg: StaticString, *extra: StaticString]():
    """Asserts that the condition must be true at compile time.

    The `constrained()` function introduces a compile-time constraint on the
    enclosing function. If the condition is true at compile time, the constraint
    has no effect. If the condition is false, compilation fails and the message
    is displayed.

    This is similar to `static_assert` in C++. It differs from
    [`debug_assert()`](/mojo/stdlib/builtin/debug_assert/debug_assert), which
    is a run-time assertion.

    Example:

    ```mojo
    fn half[dtype: DType](a: Scalar[dtype]) -> Scalar[dtype]:
        constrained[
            dtype.is_numeric(),
            "dtype must be numeric."
        ]()
        return a / 2

    def main():
        print(half(UInt8(5)))  # prints 2
        print(half(Scalar[DType.bool](True)))  # constraint failed:
                                               #     dtype must be numeric.
    ```

    Parameters:
        cond: The bool value to assert.
        msg: The message to display on failure.
        extra: Additional messages to concatenate to msg.

    """
    __mlir_op.`kgen.param.assert`[
        cond = cond.__mlir_i1__(),
        message = _get_kgen_string[msg, extra](),
    ]()


@always_inline("nodebug")
fn constrained[cond: Bool]():
    """Asserts that the condition must be true at compile time.

    The `constrained()` function introduces a compile-time constraint on the
    enclosing function. If the condition is true at compile time, the constraint
    has no effect. If the condition is false, compilation fails and a generic
    message is displayed.

    This is similar to `static_assert` in C++. It differs from
    [`debug_assert()`](/mojo/stdlib/builtin/debug_assert/debug_assert), which
    is a run-time assertion.

    For an example, see the
    [first overload](/mojo/stdlib/builtin/constrained/constrained).

    Parameters:
        cond: The bool value to assert.
    """
    constrained[cond, "param assertion failed"]()
