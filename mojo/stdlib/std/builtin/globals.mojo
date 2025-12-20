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
"""Utilities for working with global constants.

This module provides helper functions for efficiently creating references to
compile-time constants without materializing entire data structures in memory.
"""


fn global_constant[T: AnyType, //, value: T]() -> ref [StaticConstantOrigin] T:
    """Creates a reference to a compile-time constant value.

    This function uses the MLIR `pop.global_constant` operation to create a
    reference to a compile-time value without materializing the entire value
    at runtime. This is particularly useful for large lookup tables where you
    want to avoid materializing the entire table when accessing individual
    elements.

    Parameters:
        T: The type of the constant value.
        value: The compile-time constant value.

    Returns:
        A reference to the global constant.

    Examples:
    ```mojo
    from builtin.globals import global_constant

    # Create a reference to a constant array and access elements
    comptime lookup_table = InlineArray[Int, 4](1, 2, 3, 4)
    var element = global_constant[lookup_table]()[2]  # Access without materializing entire array
    print(element)  # Prints: 3

    # Use with more complex compile-time values
    fn compute(x: Int) -> Int:
        return x * 2 + 1

    comptime data = InlineArray[Int, 3](1, compute(5), 100)
    ref data_ref = global_constant[data]()
    print(data_ref[0], data_ref[1], data_ref[2])  # Prints: 1 11 100
    ```
    """
    return UnsafePointer(__mlir_op.`pop.global_constant`[value=value]())[]
