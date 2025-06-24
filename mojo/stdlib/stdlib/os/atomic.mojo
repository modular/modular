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

from collections.string.string_slice import _get_kgen_string
from sys import is_compile_time
from sys.info import is_nvidia_gpu

from builtin.dtype import _integral_type_of, _unsigned_integral_type_of
from memory import bitcast

# ===-----------------------------------------------------------------------===#
# Consistency
# ===-----------------------------------------------------------------------===#


@register_passable("trivial")
struct Consistency:
    """Represents the consistency model for atomic operations.

    The class provides a set of constants that represent different consistency
    models for atomic operations.

    Attributes:
        NOT_ATOMIC: Not atomic.
        UNORDERED: Unordered.
        MONOTONIC: Monotonic.
        ACQUIRE: Acquire.
        RELEASE: Release.
        ACQUIRE_RELEASE: Acquire-release.
        SEQUENTIAL: Sequentially consistent.
    """

    var _value: UInt8
    """The value of the consistency model.
    This is the underlying value of the consistency model.
    """

    alias NOT_ATOMIC = Self(0)
    """Not atomic."""
    alias UNORDERED = Self(1)
    """Unordered."""
    alias MONOTONIC = Self(2)
    """Monotonic."""
    alias ACQUIRE = Self(3)
    """Acquire."""
    alias RELEASE = Self(4)
    """Release."""
    alias ACQUIRE_RELEASE = Self(5)
    """Acquire-release."""
    alias SEQUENTIAL = Self(6)
    """Sequentially consistent."""

    @always_inline
    fn __init__(out self, value: UInt8):
        """Constructs a new Consistency object.

        Args:
            value: The value of the consistency model.
        """
        self._value = value

    @always_inline
    fn __eq__(self, other: Self) -> Bool:
        """Compares two Consistency objects for equality.

        Args:
            other: The other Consistency object to compare with.

        Returns:
            True if the objects are equal, False otherwise.
        """
        return self._value == other._value

    @always_inline
    fn __ne__(self, other: Self) -> Bool:
        """Compares two Consistency objects for inequality.

        Args:
            other: The other Consistency object to compare with.

        Returns:
            True if the objects are not equal, False otherwise.
        """
        return self._value != other._value

    @always_inline
    fn __is__(self, other: Self) -> Bool:
        """Checks if the Consistency object is the same as another.

        Args:
            other: The other Consistency object to compare with.

        Returns:
            True if the objects are the same, False otherwise.
        """
        return self == other

    @always_inline
    fn __isnot__(self, other: Self) -> Bool:
        """Checks if the Consistency object is not the same as another.

        Args:
            other: The other Consistency object to compare with.

        Returns:
            True if the objects are not the same, False otherwise.
        """
        return self != other

    @always_inline("nodebug")
    fn __mlir_attr(self) -> __mlir_type.`!kgen.deferred`:
        """Returns the MLIR attribute representation of the Consistency object.

        Returns:
            The MLIR attribute representation of the Consistency object.
        """
        if self is Self.NOT_ATOMIC:
            return __mlir_attr.`#pop<atomic_ordering not_atomic>`
        if self is Self.UNORDERED:
            return __mlir_attr.`#pop<atomic_ordering unordered>`
        if self is Self.MONOTONIC:
            return __mlir_attr.`#pop<atomic_ordering monotonic>`
        if self is Self.ACQUIRE:
            return __mlir_attr.`#pop<atomic_ordering acquire>`
        if self is Self.RELEASE:
            return __mlir_attr.`#pop<atomic_ordering release>`
        if self is Self.ACQUIRE_RELEASE:
            return __mlir_attr.`#pop<atomic_ordering acq_rel>`
        if self is Self.SEQUENTIAL:
            return __mlir_attr.`#pop<atomic_ordering seq_cst>`

        return abort[__mlir_type.`!kgen.deferred`]()


# ===-----------------------------------------------------------------------===#
# Atomic
# ===-----------------------------------------------------------------------===#


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

    # TODO: Unfortunate this is mut, but this is using fetch_add to load the
    # value. There is probably a better way to do this.
    @always_inline
    fn load(mut self) -> Scalar[dtype]:
        """Loads the current value from the atomic.

        Returns:
            The current value of the atomic.
        """
        return self.fetch_add(0)

    @staticmethod
    @always_inline("nodebug")
    fn fetch_add[
        *, ordering: Consistency = Consistency.SEQUENTIAL
    ](
        ptr: UnsafePointer[Scalar[dtype], mut=True, **_], rhs: Scalar[dtype]
    ) -> Scalar[dtype]:
        """Performs atomic in-place add.

        Atomically replaces the current value with the result of arithmetic
        addition of the value and arg. That is, it performs atomic
        post-increment. The operation is a read-modify-write operation. Memory
        is affected according to the value of order which is sequentially
        consistent.

        Parameters:
            ordering: The memory ordering.

        Args:
            ptr: The source pointer.
            rhs: Value to add.

        Returns:
            The original value before addition.
        """
        # Comptime interpreter doesn't support these operations.
        if is_compile_time():
            var res = ptr[]
            ptr[] += rhs
            return res

        return __mlir_op.`pop.atomic.rmw`[
            bin_op = __mlir_attr.`#pop<bin_op add>`,
            ordering = ordering.__mlir_attr(),
            syncscope = _get_kgen_string[scope](),
            _type = Scalar[dtype]._mlir_type,
        ](
            ptr.bitcast[Scalar[dtype]._mlir_type]().address,
            rhs.value,
        )

    @staticmethod
    @always_inline
    fn _xchg[
        *, ordering: Consistency = Consistency.SEQUENTIAL
    ](
        ptr: UnsafePointer[Scalar[dtype], mut=True, **_], value: Scalar[dtype]
    ) -> Scalar[dtype]:
        """Performs an atomic exchange.
        The operation is a read-modify-write operation. Memory
        is affected according to the value of order which is sequentially
        consistent.

        Parameters:
            ordering: The memory ordering.

        Args:
            ptr: The source pointer.
            value: The to exchange.

        Returns:
            The value of the value before the operation.
        """
        # Comptime interpreter doesn't support these operations.
        if is_compile_time():
            var res = ptr[]
            ptr[] = value
            return res

        return __mlir_op.`pop.atomic.rmw`[
            bin_op = __mlir_attr.`#pop<bin_op xchg>`,
            ordering = ordering.__mlir_attr(),
            _type = Scalar[dtype]._mlir_type,
        ](
            ptr.bitcast[Scalar[dtype]._mlir_type]().address,
            value.value,
        )

    @staticmethod
    @always_inline
    fn store[
        *, ordering: Consistency = Consistency.SEQUENTIAL
    ](ptr: UnsafePointer[Scalar[dtype], mut=True, **_], value: Scalar[dtype]):
        """Performs atomic store.
        The operation is a read-modify-write operation. Memory
        is affected according to the value of order which is sequentially
        consistent.

        Parameters:
            ordering: The memory ordering.

        Args:
            ptr: The source pointer.
            value: The value to store.
        """
        # Comptime interpreter doesn't support these operations.
        if is_compile_time():
            ptr[] = value
            return

        _ = __mlir_op.`pop.atomic.rmw`[
            bin_op = __mlir_attr.`#pop<bin_op xchg>`,
            ordering = ordering.__mlir_attr(),
            _type = Scalar[dtype]._mlir_type,
        ](
            ptr.bitcast[Scalar[dtype]._mlir_type]().address,
            value.value,
        )

    @always_inline
    fn fetch_add[
        *, ordering: Consistency = Consistency.SEQUENTIAL
    ](mut self, rhs: Scalar[dtype]) -> Scalar[dtype]:
        """Performs atomic in-place add.

        Atomically replaces the current value with the result of arithmetic
        addition of the value and arg. That is, it performs atomic
        post-increment. The operation is a read-modify-write operation. Memory
        is affected according to the value of order which is sequentially
        consistent.

        Parameters:
            ordering: The memory ordering.

        Args:
            rhs: Value to add.

        Returns:
            The original value before addition.
        """
        var value_addr = UnsafePointer(to=self.value)
        return Self.fetch_add[ordering=ordering](value_addr, rhs)

    @always_inline
    fn __iadd__(mut self, rhs: Scalar[dtype]):
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
    fn fetch_sub[
        *, ordering: Consistency = Consistency.SEQUENTIAL
    ](mut self, rhs: Scalar[dtype]) -> Scalar[dtype]:
        """Performs atomic in-place sub.

        Atomically replaces the current value with the result of arithmetic
        subtraction of the value and arg. That is, it performs atomic
        post-decrement. The operation is a read-modify-write operation. Memory
        is affected according to the value of order which is sequentially
        consistent.

        Parameters:
            ordering: The memory ordering.

        Args:
            rhs: Value to subtract.

        Returns:
            The original value before subtraction.
        """
        # Comptime interpreter doesn't support these operations.
        if is_compile_time():
            var res = self.value
            self.value -= rhs
            return res

        var value_addr = UnsafePointer(to=self.value.value)
        return __mlir_op.`pop.atomic.rmw`[
            bin_op = __mlir_attr.`#pop<bin_op sub>`,
            ordering = ordering.__mlir_attr(),
            syncscope = _get_kgen_string[scope](),
            _type = Scalar[dtype]._mlir_type,
        ](value_addr.address, rhs.value)

    @always_inline
    fn __isub__(mut self, rhs: Scalar[dtype]):
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
    fn compare_exchange_weak[
        *,
        failure_ordering: Consistency = Consistency.SEQUENTIAL,
        success_ordering: Consistency = Consistency.SEQUENTIAL,
    ](self, mut expected: Scalar[dtype], desired: Scalar[dtype]) -> Bool:
        """Atomically compares the self value with that of the expected value.
        If the values are equal, then the self value is replaced with the
        desired value and True is returned. Otherwise, False is returned the
        the expected value is rewritten with the self value.

        Parameters:
            failure_ordering: The memory ordering for the failure case.
            success_ordering: The memory ordering for the success case.

        Args:
          expected: The expected value.
          desired: The desired value.

        Returns:
          True if self == expected and False otherwise.
        """
        constrained[dtype.is_numeric(), "the input type must be arithmetic"]()

        @parameter
        if dtype.is_integral():
            return _compare_exchange_weak_integral_impl[
                scope=scope,
                failure_ordering=failure_ordering,
                success_ordering=success_ordering,
            ](UnsafePointer(to=self.value), expected, desired)

        # For the floating point case, we need to bitcast the floating point
        # values to their integral representation and perform the atomic
        # operation on that.

        alias integral_type = _integral_type_of[dtype]()
        var value_integral_addr = UnsafePointer(to=self.value).bitcast[
            Scalar[integral_type]
        ]()
        var expected_integral = bitcast[integral_type](expected)
        var desired_integral = bitcast[integral_type](desired)
        return _compare_exchange_weak_integral_impl[
            scope=scope,
            failure_ordering=failure_ordering,
            success_ordering=success_ordering,
        ](value_integral_addr, expected_integral, desired_integral)

    @staticmethod
    @always_inline
    fn max[
        *, ordering: Consistency = Consistency.SEQUENTIAL
    ](ptr: UnsafePointer[Scalar[dtype], **_], rhs: Scalar[dtype]):
        """Performs atomic in-place max on the pointer.

        Atomically replaces the current value pointer to by `ptr` by the result
        of max of the value and arg. The operation is a read-modify-write
        operation. The operation is a read-modify-write operation perform
        according to sequential consistency semantics.

        Constraints:
            The input type must be either integral or floating-point type.

        Parameters:
            ordering: The memory ordering.

        Args:
            ptr: The source pointer.
            rhs: Value to max.
        """
        constrained[dtype.is_numeric(), "the input type must be arithmetic"]()

        _max_impl[scope=scope, ordering=ordering](ptr, rhs)

    @always_inline
    fn max[
        *, ordering: Consistency = Consistency.SEQUENTIAL
    ](self, rhs: Scalar[dtype]):
        """Performs atomic in-place max.

        Atomically replaces the current value with the result of max of the
        value and arg. The operation is a read-modify-write operation perform
        according to sequential consistency semantics.

        Constraints:
            The input type must be either integral or floating-point type.

        Parameters:
            ordering: The memory ordering.

        Args:
            rhs: Value to max.
        """
        constrained[dtype.is_numeric(), "the input type must be arithmetic"]()

        Self.max[ordering=ordering](UnsafePointer(to=self.value), rhs)

    @staticmethod
    @always_inline
    fn min[
        *, ordering: Consistency = Consistency.SEQUENTIAL
    ](ptr: UnsafePointer[Scalar[dtype], **_], rhs: Scalar[dtype]):
        """Performs atomic in-place min on the pointer.

        Atomically replaces the current value pointer to by `ptr` by the result
        of min of the value and arg. The operation is a read-modify-write
        operation. The operation is a read-modify-write operation perform
        according to sequential consistency semantics.

        Constraints:
            The input type must be either integral or floating-point type.

        Parameters:
            ordering: The memory ordering.

        Args:
            ptr: The source pointer.
            rhs: Value to min.
        """
        constrained[dtype.is_numeric(), "the input type must be arithmetic"]()

        _min_impl[scope=scope, ordering=ordering](ptr, rhs)

    @always_inline
    fn min[
        *, ordering: Consistency = Consistency.SEQUENTIAL
    ](self, rhs: Scalar[dtype]):
        """Performs atomic in-place min.

        Atomically replaces the current value with the result of min of the
        value and arg. The operation is a read-modify-write operation. The
        operation is a read-modify-write operation perform according to
        sequential consistency semantics.

        Constraints:
            The input type must be either integral or floating-point type.

        Parameters:
            ordering: The memory ordering.

        Args:
            rhs: Value to min.
        """

        constrained[dtype.is_numeric(), "the input type must be arithmetic"]()

        Self.min[ordering=ordering](UnsafePointer(to=self.value), rhs)


# ===-----------------------------------------------------------------------===#
# Utilities
# ===-----------------------------------------------------------------------===#


@always_inline
fn _compare_exchange_weak_integral_impl[
    dtype: DType, //,
    *,
    scope: StaticString,
    failure_ordering: Consistency,
    success_ordering: Consistency,
](
    value_addr: UnsafePointer[Scalar[dtype], **_],
    mut expected: Scalar[dtype],
    desired: Scalar[dtype],
) -> Bool:
    constrained[dtype.is_integral(), "the input type must be integral"]()
    var cmpxchg_res = __mlir_op.`pop.atomic.cmpxchg`[
        failure_ordering = failure_ordering.__mlir_attr(),
        success_ordering = success_ordering.__mlir_attr(),
        syncscope = _get_kgen_string[scope](),
    ](
        value_addr.bitcast[Scalar[dtype]._mlir_type]().address,
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
    dtype: DType, //, *, scope: StaticString, ordering: Consistency
](ptr: UnsafePointer[Scalar[dtype], **_], rhs: Scalar[dtype]):
    var value_addr = ptr.bitcast[Scalar[dtype]._mlir_type]()
    _ = __mlir_op.`pop.atomic.rmw`[
        bin_op = __mlir_attr.`#pop<bin_op max>`,
        ordering = ordering.__mlir_attr(),
        syncscope = _get_kgen_string[scope](),
        _type = Scalar[dtype]._mlir_type,
    ](value_addr.address, rhs.value)


@always_inline
fn _min_impl_base[
    dtype: DType, //, *, scope: StaticString, ordering: Consistency
](ptr: UnsafePointer[Scalar[dtype], **_], rhs: Scalar[dtype]):
    var value_addr = ptr.bitcast[Scalar[dtype]._mlir_type]()
    _ = __mlir_op.`pop.atomic.rmw`[
        bin_op = __mlir_attr.`#pop<bin_op min>`,
        ordering = ordering.__mlir_attr(),
        syncscope = _get_kgen_string[scope](),
        _type = Scalar[dtype]._mlir_type,
    ](value_addr.address, rhs.value)


@always_inline
fn _max_impl[
    dtype: DType, //,
    *,
    scope: StaticString,
    ordering: Consistency,
](ptr: UnsafePointer[Scalar[dtype], **_], rhs: Scalar[dtype]):
    @parameter
    if is_nvidia_gpu() and dtype.is_floating_point():
        alias integral_type = _integral_type_of[dtype]()
        alias unsigned_integral_type = _unsigned_integral_type_of[dtype]()
        if rhs >= 0:
            _max_impl_base[scope=scope, ordering=ordering](
                ptr.bitcast[Scalar[integral_type]](),
                bitcast[integral_type](rhs),
            )
            return
        _min_impl_base[scope=scope, ordering=ordering](
            ptr.bitcast[Scalar[unsigned_integral_type]](),
            bitcast[unsigned_integral_type](rhs),
        )
        return

    _max_impl_base[scope=scope, ordering=ordering](ptr, rhs)


@always_inline
fn _min_impl[
    dtype: DType, //,
    *,
    scope: StaticString,
    ordering: Consistency,
](ptr: UnsafePointer[Scalar[dtype], **_], rhs: Scalar[dtype]):
    @parameter
    if is_nvidia_gpu() and dtype.is_floating_point():
        alias integral_type = _integral_type_of[dtype]()
        alias unsigned_integral_type = _unsigned_integral_type_of[dtype]()
        if rhs >= 0:
            _min_impl_base[scope=scope, ordering=ordering](
                ptr.bitcast[Scalar[integral_type]](),
                bitcast[integral_type](rhs),
            )
            return
        _max_impl_base[scope=scope, ordering=ordering](
            ptr.bitcast[Scalar[unsigned_integral_type]](),
            bitcast[unsigned_integral_type](rhs),
        )
        return

    _min_impl_base[scope=scope, ordering=ordering](ptr, rhs)
