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
"""Implements the `Atomic` struct.

You can import these APIs from the `os` package. For example:

```mojo
from os import Atomic
```
"""

from sys.info import is_nvidia_gpu

from builtin.dtype import _integral_type_of, _unsigned_integral_type_of
from memory import UnsafePointer, bitcast
from builtin.string_literal import get_string_literal
from collections.string import StaticString


struct Atomic[dtype: DType, *, scope: StaticString = ""]:
    """Represents a value with atomic operations.

    The class provides atomic `add` and `sub` methods for mutating the value.

    Parameters:
        dtype: DType of the value.
        scope: The memory synchronization scope.
    """

    var value: Scalar[dtype]
    """The atomic value.

    This is the underlying value of the atomic. Access to the value can only
    occur through atomic primitive operations.
    """

    @always_inline
    @implicit
    fn __init__(out self, value: Scalar[dtype]):
        """Constructs a new atomic value.

        Args:
            value: Initial value represented as `Scalar[dtype]` type.
        """
        self.value = value

    @always_inline
    fn load(self) -> Scalar[dtype]:
        """Loads the current value from the atomic.

        Returns:
            The current value of the atomic.
        """
        return self.fetch_add(0)

    @staticmethod
    @always_inline
    fn _fetch_add(
        ptr: UnsafePointer[Scalar[dtype], **_], rhs: Scalar[dtype]
    ) -> Scalar[dtype]:
        """Performs atomic in-place add.

        Atomically replaces the current value with the result of arithmetic
        addition of the value and arg. That is, it performs atomic
        post-increment. The operation is a read-modify-write operation. Memory
        is affected according to the value of order which is sequentially
        consistent.

        Args:
            ptr: The source pointer.
            rhs: Value to add.

        Returns:
            The original value before addition.
        """
        return __mlir_op.`pop.atomic.rmw`[
            bin_op = __mlir_attr.`#pop<bin_op add>`,
            ordering = __mlir_attr.`#pop<atomic_ordering seq_cst>`,
            syncscope = get_string_literal[scope]().value,
            _type = __mlir_type[`!pop.scalar<`, dtype.value, `>`],
        ](
            ptr.bitcast[
                __mlir_type[`!pop.scalar<`, dtype.value, `>`]
            ]().address,
            rhs.value,
        )

    @staticmethod
    @always_inline
    fn _xchg(
        ptr: UnsafePointer[Scalar[dtype], **_], value: Scalar[dtype]
    ) -> Scalar[dtype]:
        """Performs an atomic exchange.
        The operation is a read-modify-write operation. Memory
        is affected according to the value of order which is sequentially
        consistent.

        Args:
            ptr: The source pointer.
            value: The to exchange.

        Returns:
            The value of the value before the operation.
        """
        return __mlir_op.`pop.atomic.rmw`[
            bin_op = __mlir_attr.`#pop<bin_op xchg>`,
            ordering = __mlir_attr.`#pop<atomic_ordering seq_cst>`,
            _type = __mlir_type[`!pop.scalar<`, dtype.value, `>`],
        ](
            ptr.bitcast[
                __mlir_type[`!pop.scalar<`, dtype.value, `>`]
            ]().address,
            value.value,
        )

    @staticmethod
    @always_inline
    fn store(ptr: UnsafePointer[Scalar[dtype], **_], value: Scalar[dtype]):
        """Performs atomic store.
        The operation is a read-modify-write operation. Memory
        is affected according to the value of order which is sequentially
        consistent.

        Args:
            ptr: The source pointer.
            value: The value to store.
        """
        _ = __mlir_op.`pop.atomic.rmw`[
            bin_op = __mlir_attr.`#pop<bin_op xchg>`,
            ordering = __mlir_attr.`#pop<atomic_ordering seq_cst>`,
            _type = __mlir_type[`!pop.scalar<`, dtype.value, `>`],
        ](
            ptr.bitcast[
                __mlir_type[`!pop.scalar<`, dtype.value, `>`]
            ]().address,
            value.value,
        )

    @always_inline
    fn fetch_add(self, rhs: Scalar[dtype]) -> Scalar[dtype]:
        """Performs atomic in-place add.

        Atomically replaces the current value with the result of arithmetic
        addition of the value and arg. That is, it performs atomic
        post-increment. The operation is a read-modify-write operation. Memory
        is affected according to the value of order which is sequentially
        consistent.

        Args:
            rhs: Value to add.

        Returns:
            The original value before addition.
        """
        var value_addr = UnsafePointer.address_of(self.value)
        return Self._fetch_add(value_addr, rhs)

    @always_inline
    fn __iadd__(self, rhs: Scalar[dtype]):
        """Performs atomic in-place add.

        Atomically replaces the current value with the result of arithmetic
        addition of the value and arg. That is, it performs atomic
        post-increment. The operation is a read-modify-write operation. Memory
        is affected according to the value of order which is sequentially
        consistent.

        Args:
            rhs: Value to add.
        """
        _ = self.fetch_add(rhs)

    @always_inline
    fn fetch_sub(self, rhs: Scalar[dtype]) -> Scalar[dtype]:
        """Performs atomic in-place sub.

        Atomically replaces the current value with the result of arithmetic
        subtraction of the value and arg. That is, it performs atomic
        post-decrement. The operation is a read-modify-write operation. Memory
        is affected according to the value of order which is sequentially
        consistent.

        Args:
            rhs: Value to subtract.

        Returns:
            The original value before subtraction.
        """
        var value_addr = UnsafePointer.address_of(self.value.value)
        return __mlir_op.`pop.atomic.rmw`[
            bin_op = __mlir_attr.`#pop<bin_op sub>`,
            ordering = __mlir_attr.`#pop<atomic_ordering seq_cst>`,
            syncscope = get_string_literal[scope]().value,
            _type = __mlir_type[`!pop.scalar<`, dtype.value, `>`],
        ](value_addr.address, rhs.value)

    @always_inline
    fn __isub__(self, rhs: Scalar[dtype]):
        """Performs atomic in-place sub.

        Atomically replaces the current value with the result of arithmetic
        subtraction of the value and arg. That is, it performs atomic
        post-decrement. The operation is a read-modify-write operation. Memory
        is affected according to the value of order which is sequentially
        consistent.

        Args:
            rhs: Value to subtract.
        """
        _ = self.fetch_sub(rhs)

    @always_inline
    fn compare_exchange_weak(
        self, mut expected: Scalar[dtype], desired: Scalar[dtype]
    ) -> Bool:
        """Atomically compares the self value with that of the expected value.
        If the values are equal, then the self value is replaced with the
        desired value and True is returned. Otherwise, False is returned the
        the expected value is rewritten with the self value.

        Args:
          expected: The expected value.
          desired: The desired value.

        Returns:
          True if self == expected and False otherwise.
        """
        constrained[dtype.is_numeric(), "the input type must be arithmetic"]()

        @parameter
        if dtype.is_integral():
            return _compare_exchange_weak_integral_impl[scope=scope](
                UnsafePointer.address_of(self.value), expected, desired
            )

        # For the floating point case, we need to bitcast the floating point
        # values to their integral representation and perform the atomic
        # operation on that.

        alias integral_type = _integral_type_of[dtype]()
        var value_integral_addr = UnsafePointer.address_of(self.value).bitcast[
            Scalar[integral_type]
        ]()
        var expected_integral = bitcast[integral_type](expected)
        var desired_integral = bitcast[integral_type](desired)
        return _compare_exchange_weak_integral_impl[scope=scope](
            value_integral_addr, expected_integral, desired_integral
        )

    @staticmethod
    @always_inline
    fn max(ptr: UnsafePointer[Scalar[dtype], **_], rhs: Scalar[dtype]):
        """Performs atomic in-place max on the pointer.

        Atomically replaces the current value pointer to by `ptr` by the result
        of max of the value and arg. The operation is a read-modify-write
        operation. The operation is a read-modify-write operation perform
        according to sequential consistency semantics.

        Constraints:
            The input type must be either integral or floating-point type.

        Args:
            ptr: The source pointer.
            rhs: Value to max.
        """
        constrained[dtype.is_numeric(), "the input type must be arithmetic"]()

        _max_impl[scope=scope](ptr, rhs)

    @always_inline
    fn max(self, rhs: Scalar[dtype]):
        """Performs atomic in-place max.

        Atomically replaces the current value with the result of max of the
        value and arg. The operation is a read-modify-write operation perform
        according to sequential consistency semantics.

        Constraints:
            The input type must be either integral or floating-point type.


        Args:
            rhs: Value to max.
        """
        constrained[dtype.is_numeric(), "the input type must be arithmetic"]()

        Self.max(UnsafePointer.address_of(self.value), rhs)

    @staticmethod
    @always_inline
    fn min(ptr: UnsafePointer[Scalar[dtype], **_], rhs: Scalar[dtype]):
        """Performs atomic in-place min on the pointer.

        Atomically replaces the current value pointer to by `ptr` by the result
        of min of the value and arg. The operation is a read-modify-write
        operation. The operation is a read-modify-write operation perform
        according to sequential consistency semantics.

        Constraints:
            The input type must be either integral or floating-point type.

        Args:
            ptr: The source pointer.
            rhs: Value to min.
        """
        constrained[dtype.is_numeric(), "the input type must be arithmetic"]()

        _min_impl[scope=scope](ptr, rhs)

    @always_inline
    fn min(self, rhs: Scalar[dtype]):
        """Performs atomic in-place min.

        Atomically replaces the current value with the result of min of the
        value and arg. The operation is a read-modify-write operation. The
        operation is a read-modify-write operation perform according to
        sequential consistency semantics.

        Constraints:
            The input type must be either integral or floating-point type.

        Args:
            rhs: Value to min.
        """

        constrained[dtype.is_numeric(), "the input type must be arithmetic"]()

        Self.min(UnsafePointer.address_of(self.value), rhs)


# ===-----------------------------------------------------------------------===#
# Utilities
# ===-----------------------------------------------------------------------===#


@always_inline
fn _compare_exchange_weak_integral_impl[
    dtype: DType, //, *, scope: StaticString
](
    value_addr: UnsafePointer[Scalar[dtype], **_],
    mut expected: Scalar[dtype],
    desired: Scalar[dtype],
) -> Bool:
    constrained[dtype.is_integral(), "the input type must be integral"]()
    var cmpxchg_res = __mlir_op.`pop.atomic.cmpxchg`[
        failure_ordering = __mlir_attr.`#pop<atomic_ordering seq_cst>`,
        success_ordering = __mlir_attr.`#pop<atomic_ordering seq_cst>`,
        syncscope = get_string_literal[scope]().value,
    ](
        value_addr.bitcast[
            __mlir_type[`!pop.scalar<`, dtype.value, `>`]
        ]().address,
        expected.value,
        desired.value,
    )
    var ok = Bool(
        __mlir_op.`kgen.struct.extract`[index = __mlir_attr.`1:index`](
            cmpxchg_res
        )
    )
    if not ok:
        expected = value_addr[]
    return ok


@always_inline
fn _max_impl_base[
    dtype: DType, //, *, scope: StaticString
](ptr: UnsafePointer[Scalar[dtype], **_], rhs: Scalar[dtype]):
    var value_addr = ptr.bitcast[
        __mlir_type[`!pop.scalar<`, dtype.value, `>`]
    ]()
    _ = __mlir_op.`pop.atomic.rmw`[
        bin_op = __mlir_attr.`#pop<bin_op max>`,
        ordering = __mlir_attr.`#pop<atomic_ordering seq_cst>`,
        syncscope = get_string_literal[scope]().value,
        _type = __mlir_type[`!pop.scalar<`, dtype.value, `>`],
    ](value_addr.address, rhs.value)


@always_inline
fn _min_impl_base[
    dtype: DType, //, *, scope: StaticString
](ptr: UnsafePointer[Scalar[dtype], **_], rhs: Scalar[dtype]):
    var value_addr = ptr.bitcast[
        __mlir_type[`!pop.scalar<`, dtype.value, `>`]
    ]()
    _ = __mlir_op.`pop.atomic.rmw`[
        bin_op = __mlir_attr.`#pop<bin_op min>`,
        ordering = __mlir_attr.`#pop<atomic_ordering seq_cst>`,
        syncscope = get_string_literal[scope]().value,
        _type = __mlir_type[`!pop.scalar<`, dtype.value, `>`],
    ](value_addr.address, rhs.value)


@always_inline
fn _max_impl[
    dtype: DType, //, *, scope: StaticString
](ptr: UnsafePointer[Scalar[dtype], **_], rhs: Scalar[dtype]):
    @parameter
    if is_nvidia_gpu() and dtype.is_floating_point():
        alias integral_type = _integral_type_of[dtype]()
        alias unsigned_integral_type = _unsigned_integral_type_of[dtype]()
        if rhs >= 0:
            _max_impl_base[scope=scope](
                ptr.bitcast[Scalar[integral_type]](),
                bitcast[integral_type](rhs),
            )
            return
        _min_impl_base[scope=scope](
            ptr.bitcast[Scalar[unsigned_integral_type]](),
            bitcast[unsigned_integral_type](rhs),
        )
        return

    _max_impl_base[scope=scope](ptr, rhs)


@always_inline
fn _min_impl[
    dtype: DType, //, *, scope: StaticString
](ptr: UnsafePointer[Scalar[dtype], **_], rhs: Scalar[dtype]):
    @parameter
    if is_nvidia_gpu() and dtype.is_floating_point():
        alias integral_type = _integral_type_of[dtype]()
        alias unsigned_integral_type = _unsigned_integral_type_of[dtype]()
        if rhs >= 0:
            _min_impl_base[scope=scope](
                ptr.bitcast[Scalar[integral_type]](),
                bitcast[integral_type](rhs),
            )
            return
        _max_impl_base[scope=scope](
            ptr.bitcast[Scalar[unsigned_integral_type]](),
            bitcast[unsigned_integral_type](rhs),
        )
        return

    _min_impl_base[scope=scope](ptr, rhs)
