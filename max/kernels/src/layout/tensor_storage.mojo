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
"""Defines storage abstractions for tile-backed tensor views."""


from std.builtin.device_passable import DevicePassable
from std.builtin.int import index
from std.gpu.host import DevicePointer
from std.os import abort
from std.sys import size_of
from std.sys.info import is_gpu
from layout import TensorLayout, Idx


trait TensorStorage:
    """Defines a non-owning interface for accessing tensor storage.

    A conforming type describes how to access storage that is owned elsewhere.
    It provides a concrete `StorageType` handle along with static operations to
    load from, store to, offset into, and reinterpret values of that handle. The
    trait never owns the underlying memory; the handle's `origin` parameter
    tracks the lifetime and mutability of the borrowed storage.
    """

    comptime element_size: Int = 1

    comptime StorageType[
        mut: Bool,
        //,
        dtype: DType,
        origin: Origin[mut=mut],
        address_space: AddressSpace,
    ]: TrivialRegisterPassable
    """The concrete, register-passable handle to the borrowed storage.

    Every operation in this trait acts on values of this type. It is
    parameterized on the element `dtype`, the `origin` that tracks the lifetime
    and mutability of the borrowed storage, and the `address_space` the storage
    resides in, so a single conforming type describes a whole family of handles.

    Parameters:
        mut: The mutability of the borrowed storage, inferred from `origin`.
        dtype: The element data type of the borrowed storage.
        origin: The origin tracking the lifetime of the borrowed storage.
        address_space: The address space the borrowed storage resides in.
    """

    @staticmethod
    def write_type_name_to(mut writer: Some[Writer]):
        """Write the storage type name representation to the writer.

        Args:
            writer: The `Writer` to output to.
        """
        reflect[Self].name().write_to(writer)

    @doc_hidden
    @staticmethod
    def unsafe_ptr[
        mut: Bool,
        dtype: DType,
        origin: Origin[mut=mut],
        address_space: AddressSpace,
        //,
    ](
        storage: Self.StorageType[dtype, origin, address_space],
    ) raises -> UnsafePointer[
        Scalar[dtype], origin, address_space=address_space
    ]:
        """Returns a raw scalar pointer to the borrowed storage.

        Reinterprets the storage handle as an `UnsafePointer` to the scalar
        base of the referenced storage; no conversion of the stored elements
        takes place. The returned pointer borrows the same externally owned
        memory that the handle refers to; the trait still does not own it.

        Parameters:
            mut: The mutability of the borrowed storage, inferred from `origin`.
            dtype: The element data type of the borrowed storage.
            origin: The origin tracking the lifetime of the borrowed storage.
            address_space: The address space the borrowed storage resides in.

        Args:
            storage: The storage to reinterpret as a raw scalar pointer.

        Returns:
            An `UnsafePointer` to `Scalar[dtype]` referring to the base of the
            borrowed storage.

        Raises:
            An error if the backing storage does not support accessing a
            pointer to the underlying data.
        """
        ...

    @staticmethod
    def unsafe_cast[
        to_mut: Bool,
        //,
        to_dtype: DType,
        to_origin: Origin[mut=to_mut],
        to_address_space: AddressSpace,
    ](storage: Self.StorageType[...]) -> Self.StorageType[
        to_dtype, to_origin, to_address_space
    ]:
        """Reinterprets a storage handle with new type parameters.

        This performs an unchecked reinterpretation of the underlying reference;
        no conversion of the stored elements takes place. The caller is
        responsible for ensuring the new `dtype`, `origin`, and `address_space`
        are valid for the referenced storage.

        Parameters:
            to_mut: The mutability to reinterpret the storage as.
            to_dtype: The element data type to reinterpret the storage as.
            to_origin: The origin to reinterpret the storage as.
            to_address_space: The address space to reinterpret the storage as.

        Args:
            storage: The storage to reinterpret.

        Returns:
            A handle referring to the same storage, viewed with the new type
            parameters.
        """
        ...

    @staticmethod
    def load[
        dtype: DType,
        //,
        width: SIMDSize,
        alignment: Int,
        invariant: Bool = False,
        non_temporal: Bool = False,
    ](storage: Self.StorageType[mut=False, dtype, ...]) -> SIMD[dtype, width]:
        """Loads a `SIMD` value from the storage.

        Parameters:
            dtype: The element data type of the storage.
            width: The number of elements to load.
            alignment: The alignment guarantee for the load.
            invariant: If True, the compiler may assume the memory won't be
                modified during the kernel, enabling load hoisting and caching.
            non_temporal: If True, indicates the data will not be reused soon,
                allowing the hardware to bypass caches (e.g., streaming loads).

        Args:
            storage: The storage to load from.

        Returns:
            The loaded `SIMD` value.
        """
        ...

    @staticmethod
    def load[
        dtype: DType,
        //,
        width: SIMDSize,
        alignment: Int,
        invariant: Bool = False,
        non_temporal: Bool = False,
    ](
        storage: Self.StorageType[mut=False, dtype, ...],
        offset: Some[Indexer],
    ) -> SIMD[dtype, width]:
        """Loads a `SIMD` value at a scalar-element offset from the storage.

        Parameters:
            dtype: The element data type of the storage.
            width: The number of elements to load.
            alignment: The alignment guarantee for the load.
            invariant: If True, the compiler may assume the memory won't be
                modified during the kernel, enabling load hoisting and caching.
            non_temporal: If True, indicates the data will not be reused soon,
                allowing the hardware to bypass caches (e.g., streaming loads).

        Args:
            storage: The storage to load from.
            offset: The scalar-element offset to load at.

        Returns:
            The loaded `SIMD` value.
        """
        ...

    @staticmethod
    def store[
        dtype: DType,
        alignment: Int,
        *,
        non_temporal: Bool = False,
    ](storage: Self.StorageType[mut=True, dtype, ...], value: SIMD[dtype, _],):
        """Stores a `SIMD` value into the storage.

        Parameters:
            dtype: The element data type of the storage.
            alignment: The alignment guarantee for the store.
            non_temporal: If True, indicates the data will not be reused soon,
                allowing the hardware to bypass caches (e.g., streaming stores).

        Args:
            storage: The storage to store into.
            value: The `SIMD` value to store.
        """
        ...

    @staticmethod
    def store[
        dtype: DType,
        alignment: Int,
        *,
        non_temporal: Bool = False,
    ](
        storage: Self.StorageType[mut=True, dtype, ...],
        offset: Some[Indexer],
        value: SIMD[dtype, _],
    ):
        """Stores a `SIMD` value at a scalar-element offset in the storage.

        The caller is responsible for ensuring the storage is actually mutable.
        The `dtype`, `origin`, and `address_space` are inferred from the
        `storage` argument for concrete storage types; callers using the trait
        through an abstract `TensorStorage` bound must pass them explicitly
        (before `alignment`).

        Parameters:
            dtype: The element data type of the storage.
            alignment: The alignment guarantee for the store.
            non_temporal: If True, indicates the data will not be reused soon,
                allowing the hardware to bypass caches (e.g., streaming stores).

        Args:
            storage: The storage to store into.
            offset: The scalar-element offset to store at.
            value: The `SIMD` value to store.
        """
        ...

    @staticmethod
    def offset(
        storage: Self.StorageType[...], offset: Some[Indexer]
    ) -> type_of(storage):
        """Returns a storage handle offset by a number of scalar elements.

        Args:
            storage: The storage to offset from.
            offset: The number of scalar elements to advance the handle by.

        Returns:
            A handle of the same type starting `offset` scalar elements into
            the referenced storage.
        """
        ...

    @staticmethod
    def distance[
        dtype: DType, address_space: AddressSpace, //
    ](
        storage: Self.StorageType[mut=False, dtype, _, address_space],
        other: Self.StorageType[mut=False, dtype, _, address_space],
    ) -> Int:
        """Returns the scalar-element distance from `other` to `storage`.

        Parameters:
            dtype: The storages' `DType`.
            address_space: The storages' `AddressSpace`.

        Args:
            storage: The storage to measure the distance to.
            other: The storage to measure the distance from.

        Returns:
            The number of scalar elements separating the two handles. The
            value is positive when `storage` is ahead of `other` and negative
            when it precedes `other`.
        """
        ...


trait TensorArith(TensorStorage):
    """Extends `TensorStorage` with in-place elementwise arithmetic.

    A conforming type provides the same non-owning storage handle as
    `TensorStorage`, plus a family of in-place elementwise binary operations.
    Each operation takes its operands as `(storage, layout)` tuples, so the
    layout describing a handle travels alongside it.
    """

    @staticmethod
    def add[
        SelfLayoutType: TensorLayout,
        self_origin: MutOrigin,
        self_address_space: AddressSpace,
        OtherLayoutType: TensorLayout,
        other_mut: Bool,
        other_origin: Origin[mut=other_mut],
        other_address_space: AddressSpace,
        //,
        dtype: DType,
    ](
        storage: Tuple[
            Self.StorageType[dtype, self_origin, self_address_space],
            SelfLayoutType,
        ],
        other: Tuple[
            Self.StorageType[dtype, other_origin, other_address_space],
            OtherLayoutType,
        ],
    ):
        """Adds `other` into `storage` elementwise, in place.

        Parameters:
            SelfLayoutType: The layout type of the destination storage.
            self_origin: The origin of the destination storage.
            self_address_space: The address space of the destination storage.
            OtherLayoutType: The layout type of the right-hand storage operand.
            other_mut: The mutability of the right-hand storage operand.
            other_origin: The origin of the right-hand storage operand.
            other_address_space: The address space of the right-hand storage
                operand.
            dtype: The element data type of both storages.

        Args:
            storage: A tuple of the destination storage (modified in place) and
                its layout.
            other: A tuple of the right-hand storage operand and its layout.
        """
        ...

    @staticmethod
    def mul[
        SelfLayoutType: TensorLayout,
        self_origin: MutOrigin,
        self_address_space: AddressSpace,
        OtherLayoutType: TensorLayout,
        other_mut: Bool,
        other_origin: Origin[mut=other_mut],
        other_address_space: AddressSpace,
        //,
        dtype: DType,
    ](
        storage: Tuple[
            Self.StorageType[dtype, self_origin, self_address_space],
            SelfLayoutType,
        ],
        other: Tuple[
            Self.StorageType[dtype, other_origin, other_address_space],
            OtherLayoutType,
        ],
    ):
        """Multiplies `storage` by `other` elementwise, in place.

        Parameters:
            SelfLayoutType: The layout type of the destination storage.
            self_origin: The origin of the destination storage.
            self_address_space: The address space of the destination storage.
            OtherLayoutType: The layout type of the right-hand storage operand.
            other_mut: The mutability of the right-hand storage operand.
            other_origin: The origin of the right-hand storage operand.
            other_address_space: The address space of the right-hand storage
                operand.
            dtype: The element data type of both storages.

        Args:
            storage: A tuple of the destination storage (modified in place) and
                its layout.
            other: A tuple of the right-hand storage operand and its layout.
        """
        ...

    @staticmethod
    def sub[
        SelfLayoutType: TensorLayout,
        self_origin: MutOrigin,
        self_address_space: AddressSpace,
        OtherLayoutType: TensorLayout,
        other_mut: Bool,
        other_origin: Origin[mut=other_mut],
        other_address_space: AddressSpace,
        //,
        dtype: DType,
    ](
        storage: Tuple[
            Self.StorageType[dtype, self_origin, self_address_space],
            SelfLayoutType,
        ],
        other: Tuple[
            Self.StorageType[dtype, other_origin, other_address_space],
            OtherLayoutType,
        ],
    ):
        """Subtracts `other` from `storage` elementwise, in place.

        Parameters:
            SelfLayoutType: The layout type of the destination storage.
            self_origin: The origin of the destination storage.
            self_address_space: The address space of the destination storage.
            OtherLayoutType: The layout type of the right-hand storage operand.
            other_mut: The mutability of the right-hand storage operand.
            other_origin: The origin of the right-hand storage operand.
            other_address_space: The address space of the right-hand storage
                operand.
            dtype: The element data type of both storages.

        Args:
            storage: A tuple of the destination storage (modified in place) and
                its layout.
            other: A tuple of the right-hand storage operand and its layout.
        """
        ...

    @staticmethod
    def floordiv[
        SelfLayoutType: TensorLayout,
        self_origin: MutOrigin,
        self_address_space: AddressSpace,
        OtherLayoutType: TensorLayout,
        other_mut: Bool,
        other_origin: Origin[mut=other_mut],
        other_address_space: AddressSpace,
        //,
        dtype: DType,
    ](
        storage: Tuple[
            Self.StorageType[dtype, self_origin, self_address_space],
            SelfLayoutType,
        ],
        other: Tuple[
            Self.StorageType[dtype, other_origin, other_address_space],
            OtherLayoutType,
        ],
    ):
        """Floor-divides `storage` by `other` elementwise, in place.

        Parameters:
            SelfLayoutType: The layout type of the destination storage.
            self_origin: The origin of the destination storage.
            self_address_space: The address space of the destination storage.
            OtherLayoutType: The layout type of the right-hand storage operand.
            other_mut: The mutability of the right-hand storage operand.
            other_origin: The origin of the right-hand storage operand.
            other_address_space: The address space of the right-hand storage
                operand.
            dtype: The element data type of both storages.

        Args:
            storage: A tuple of the destination storage (modified in place) and
                its layout.
            other: A tuple of the right-hand storage operand and its layout.
        """
        ...

    @staticmethod
    def truediv[
        SelfLayoutType: TensorLayout,
        self_origin: MutOrigin,
        self_address_space: AddressSpace,
        OtherLayoutType: TensorLayout,
        other_mut: Bool,
        other_origin: Origin[mut=other_mut],
        other_address_space: AddressSpace,
        //,
        dtype: DType,
    ](
        storage: Tuple[
            Self.StorageType[dtype, self_origin, self_address_space],
            SelfLayoutType,
        ],
        other: Tuple[
            Self.StorageType[dtype, other_origin, other_address_space],
            OtherLayoutType,
        ],
    ):
        """True-divides `storage` by `other` elementwise, in place.

        Parameters:
            SelfLayoutType: The layout type of the destination storage.
            self_origin: The origin of the destination storage.
            self_address_space: The address space of the destination storage.
            OtherLayoutType: The layout type of the right-hand storage operand.
            other_mut: The mutability of the right-hand storage operand.
            other_origin: The origin of the right-hand storage operand.
            other_address_space: The address space of the right-hand storage
                operand.
            dtype: The element data type of both storages.

        Args:
            storage: A tuple of the destination storage (modified in place) and
                its layout.
            other: A tuple of the right-hand storage operand and its layout.
        """
        ...


struct PointerStorage[*, element_width: Int = 1](TensorArith):
    """Implements `TensorArith` backed by a raw `UnsafePointer`.

    `PointerStorage` is the default storage policy for `TileTensor`. Its
    `StorageType` handle is a plain `UnsafePointer`, and every operation is
    expressed directly in terms of the underlying pointer.

    Parameters:
        element_width: Number of scalar elements per logical element. A value
            of `1` (the default) is a non-vectorized tensor; larger values
            describe a vectorized view whose logical elements are SIMD vectors.
    """

    comptime element_size = Self.element_width
    """Number of scalar elements per logical element (alias of `element_width`)."""

    comptime StorageType[
        mut: Bool,
        //,
        dtype: DType,
        origin: Origin[mut=mut],
        address_space: AddressSpace,
    ]: TrivialRegisterPassable = UnsafePointer[
        SIMD[dtype, Self.element_width], origin, address_space=address_space
    ]
    """A raw `UnsafePointer` to `Scalar[dtype]` borrowing the storage.

    Parameters:
        mut: The mutability of the borrowed storage, inferred from `origin`.
        dtype: The element data type of the borrowed storage.
        origin: The origin tracking the lifetime of the borrowed storage.
        address_space: The address space the borrowed storage resides in.
    """

    @staticmethod
    def write_type_name_to(mut writer: Some[Writer]):
        """Write the storage type name representation to the writer.

        Args:
            writer: The `Writer` to output to.
        """
        t"PointerStorage[element_size={Self.element_size}]".write_to(writer)

    @doc_hidden
    @staticmethod
    @always_inline
    def unsafe_ptr[
        mut: Bool,
        dtype: DType,
        origin: Origin[mut=mut],
        address_space: AddressSpace,
        //,
    ](
        storage: Self.StorageType[dtype, origin, address_space],
    ) raises -> UnsafePointer[
        Scalar[dtype], origin, address_space=address_space
    ]:
        """Returns a raw scalar pointer to the borrowed storage.

        Parameters:
            mut: The mutability of the borrowed storage, inferred from `origin`.
            dtype: The element data type of the borrowed storage.
            origin: The origin tracking the lifetime of the borrowed storage.
            address_space: The address space the borrowed storage resides in.

        Args:
            storage: The storage to reinterpret as a raw scalar pointer.

        Returns:
            An `UnsafePointer` to `Scalar[dtype]` referring to the base of the
            borrowed storage.
        """
        # `storage` is an `UnsafePointer[SIMD[dtype, element_width]]`. Bitcast
        # it to the scalar base pointer. For non-vectorized storage
        # (`element_width == 1`) this is the identity; for a vectorized view it
        # yields the scalar base address of the underlying storage.
        return storage.bitcast[Scalar[dtype]]()

    @staticmethod
    @always_inline
    def unsafe_cast[
        to_mut: Bool,
        //,
        to_dtype: DType,
        to_origin: Origin[mut=to_mut],
        to_address_space: AddressSpace,
    ](
        storage: Self.StorageType[...],
        out result: Self.StorageType[
            mut=to_mut, to_dtype, to_origin, to_address_space
        ],
    ):
        """Reinterprets a storage handle with new type parameters.

        Parameters:
            to_mut: The mutability of the origin.
            to_dtype: The element data type to reinterpret the storage as.
            to_origin: The origin to reinterpret the storage as.
            to_address_space: The address space to reinterpret the storage as.

        Args:
            storage: The storage to reinterpret.

        Returns:
            A handle referring to the same storage, viewed with the new type
            parameters.
        """
        result = {
            _mlir_value = __mlir_op.`pop.pointer.bitcast`[
                _type=type_of(result)._mlir_type,
            ](storage._get_kgen_pointer())
        }

    @staticmethod
    @always_inline
    def load[
        dtype: DType,
        //,
        width: SIMDSize,
        alignment: Int,
        invariant: Bool = False,
        non_temporal: Bool = False,
    ](storage: Self.StorageType[mut=False, dtype, ...]) -> SIMD[dtype, width]:
        """Loads a `SIMD` value from the storage.

        Parameters:
            dtype: The element data type of the storage.
            width: The number of elements to load.
            alignment: The alignment guarantee for the load.
            invariant: If True, the compiler may assume the memory won't be
                modified during the kernel, enabling load hoisting and caching.
            non_temporal: If True, indicates the data will not be reused soon,
                allowing the hardware to bypass caches (e.g., streaming loads).

        Args:
            storage: The storage to load from.

        Returns:
            The loaded `SIMD` value.
        """
        return storage.bitcast[Scalar[dtype]]().load[
            width=width,
            alignment=alignment,
            invariant=invariant,
            non_temporal=non_temporal,
        ]()

    @staticmethod
    @always_inline
    def load[
        dtype: DType,
        //,
        width: SIMDSize,
        alignment: Int,
        invariant: Bool = False,
        non_temporal: Bool = False,
    ](
        storage: Self.StorageType[mut=False, dtype, ...],
        offset: Some[Indexer],
    ) -> SIMD[dtype, width]:
        """Loads a `SIMD` value at a scalar-element offset from the storage.

        Parameters:
            dtype: The element data type of the storage.
            width: The number of elements to load.
            alignment: The alignment guarantee for the load.
            invariant: If True, the compiler may assume the memory won't be
                modified during the kernel, enabling load hoisting and caching.
            non_temporal: If True, indicates the data will not be reused soon,
                allowing the hardware to bypass caches (e.g., streaming loads).

        Args:
            storage: The storage to load from.
            offset: The scalar-element offset to load at.

        Returns:
            The loaded `SIMD` value.
        """
        return storage.bitcast[Scalar[dtype]]().load[
            width=width,
            alignment=alignment,
            invariant=invariant,
            non_temporal=non_temporal,
        ](offset)

    @staticmethod
    @always_inline
    def store[
        dtype: DType,
        alignment: Int,
        *,
        non_temporal: Bool = False,
    ](storage: Self.StorageType[mut=True, dtype, ...], value: SIMD[dtype, _],):
        """Stores a `SIMD` value into the storage.

        Parameters:
            dtype: The element data type of the storage.
            alignment: The alignment guarantee for the store.
            non_temporal: If True, indicates the data will not be reused soon,
                allowing the hardware to bypass caches (e.g., streaming stores).

        Args:
            storage: The storage to store into.
            value: The `SIMD` value to store.
        """
        storage.bitcast[Scalar[dtype]]().store[
            alignment=alignment, non_temporal=non_temporal
        ](value)

    @staticmethod
    @always_inline
    def store[
        dtype: DType,
        alignment: Int,
        *,
        non_temporal: Bool = False,
    ](
        storage: Self.StorageType[mut=True, dtype, ...],
        offset: Some[Indexer],
        value: SIMD[dtype, _],
    ):
        """Stores a `SIMD` value at a scalar-element offset in the storage.

        Parameters:
            dtype: The element data type of the storage.
            alignment: The alignment guarantee for the store.
            non_temporal: If True, indicates the data will not be reused soon,
                allowing the hardware to bypass caches (e.g., streaming stores).

        Args:
            storage: The storage to store into.
            offset: The scalar-element offset to store at.
            value: The `SIMD` value to store.
        """
        storage.bitcast[Scalar[dtype]]().store[
            alignment=alignment, non_temporal=non_temporal
        ](offset, value)

    @staticmethod
    @always_inline
    def offset(
        storage: Self.StorageType[...], offset: Some[Indexer]
    ) -> type_of(storage):
        """Returns a storage handle offset by a number of scalar elements.

        The returned handle refers to the same externally owned storage,
        advanced by `offset` scalar elements. The offset is measured in scalar
        elements (not logical SIMD elements) so that it matches the scalar-unit
        offsets produced by a tensor's layout and consumed by `load`/`store`;
        for a vectorized storage (`element_width > 1`) advancing the raw
        SIMD-typed handle directly would over-advance by `element_width`.

        Args:
            storage: The storage to offset from.
            offset: The number of scalar elements to advance the handle by.

        Returns:
            A handle of the same type starting `offset` scalar elements into
            the referenced storage.
        """
        # `storage` is an `UnsafePointer[SIMD[dtype, element_width]]`. Reinterpret
        # it as a scalar pointer so `+ offset` advances in scalar (not SIMD)
        # units, then `rebind` back to the original handle type.
        return (
            storage.bitcast[Scalar[type_of(storage).type.dtype]]() + offset
        ).bitcast[SIMD[type_of(storage).type.dtype, Self.element_width]]()

    @staticmethod
    def distance[
        dtype: DType, address_space: AddressSpace, //
    ](
        storage: Self.StorageType[mut=False, dtype, _, address_space],
        other: Self.StorageType[mut=False, dtype, _, address_space],
    ) -> Int:
        """Returns the scalar-element distance from `other` to `storage`.

        Parameters:
            dtype: The storages' `DType`.
            address_space: The storages' `AddressSpace`.

        Args:
            storage: The storage to measure the distance to.
            other: The storage to measure the distance from.

        Returns:
            The number of scalar elements separating the two handles. The
            value is positive when `storage` is ahead of `other` and negative
            when it precedes `other`.
        """
        return (Int(storage) - Int(other)) // size_of[dtype]()

    @always_inline
    @staticmethod
    def _elementwise_binary_with_broadcast[
        SelfLayoutType: TensorLayout,
        self_origin: MutOrigin,
        self_address_space: AddressSpace,
        OtherLayoutType: TensorLayout,
        other_mut: Bool,
        other_origin: Origin[mut=other_mut],
        other_address_space: AddressSpace,
        //,
        dtype: DType,
    ](
        storage: Tuple[
            Self.StorageType[dtype, self_origin, self_address_space],
            SelfLayoutType,
        ],
        other: Tuple[
            Self.StorageType[dtype, other_origin, other_address_space],
            OtherLayoutType,
        ],
        func: Some[
            def(
                SIMD[dtype, Self.element_width], SIMD[dtype, Self.element_width]
            ) -> (SIMD[dtype, Self.element_width])
        ],
    ):
        """Apply an elementwise binary operation with broadcasting support.

        This internal method applies a binary operation between elements of this
        tensor and another tensor, with support for limited broadcasting
        patterns. The operation is performed in-place on this tensor.

        Parameters:
            SelfLayoutType: The layout type of the destination storage.
            self_origin: The origin of the destination storage.
            self_address_space: The address space of the destination storage.
            OtherLayoutType: The layout type of the right-hand storage operand.
            other_mut: The mutability of the right-hand storage operand.
            other_origin: The origin of the right-hand storage operand.
            other_address_space: The address space of the right-hand storage
                operand.
            dtype: The dtype of both tensors's elements.

        Args:
            storage: A tuple of the destination storage (modified in place) and
                its layout.
            other: A tuple of the right-hand storage operand and its layout.
            func: A binary function that takes two elements (one from each
                tensor) and returns a single element as the result of the
                operation.

        Notes:

        - Currently supports only rank-2 tensors or tensors of the same rank.
        - For tensors of the same rank, shapes must match exactly.
        - For rank-1 to rank-2 broadcasting, the rank-1 tensor's dimension must
            match the corresponding dimension of the rank-2 tensor.
        - The operation is optimized based on the memory layout of both tensors.
        """

        ref self_storage = storage[0]
        ref self_layout = storage[1]
        ref other_storage = other[0]
        ref other_layout = other[1]

        comptime self_rank = type_of(self_layout).rank
        comptime other_rank = type_of(other_layout).rank
        comptime other_shape[i: Int] = type_of(other_layout).static_shape[i]
        comptime self_shape[i: Int] = type_of(self_layout).static_shape[i]

        comptime if self_rank == other_rank:
            comptime for axis in range(type_of(self_layout).rank):
                comptime assert other_shape[axis] == self_shape[axis], (
                    "_elementwise_binary_with_broadcast requires shape to"
                    " be the same for tensors of the same rank"
                )

        comptime assert type_of(self_layout).all_dims_known, (
            "_elementwise_binary_with_broadcast must operates on tensors"
            " of statically know layouts"
        )
        comptime assert other_rank <= self_rank, (
            "_elementwise_binary_with_broadcast must operates on tensor of"
            " equal of lower rank"
        )

        # TODO(KERN-812): Support numpy like broadcasting and relax rank-2
        # constrain.
        comptime assert (
            self_rank == 2 or self_rank == other_rank
        ), "Only supports rank-2 tensor, or same rank"

        comptime if other_rank == 1:
            comptime assert other_shape[0] == self_shape[0], (
                "_elementwise_binary_with_broadcast 1d tensor operand must"
                " have a dim that matches the tensors"
            )

            comptime for i in range(type_of(self_layout).static_product):
                comptime other_size = type_of(other_layout).static_product

                var lhs_idx = self_layout(Idx[i])
                var rhs_idx = other_layout(Idx[i % other_size])

                self_storage.bitcast[Scalar[dtype]]().store(
                    lhs_idx,
                    func(
                        self_storage.bitcast[Scalar[dtype]]().load[
                            width=Self.element_width
                        ](lhs_idx),
                        other_storage.bitcast[Scalar[dtype]]().load[
                            width=Self.element_width
                        ](rhs_idx),
                    ),
                )

        comptime for i in range(type_of(self_layout).static_product):
            var lhs_idx = self_layout(Idx[i])
            var rhs_idx = other_layout(Idx[i])
            self_storage.bitcast[Scalar[dtype]]().store(
                lhs_idx,
                func(
                    self_storage.bitcast[Scalar[dtype]]().load[
                        width=Self.element_size
                    ](lhs_idx),
                    other_storage.bitcast[Scalar[dtype]]().load[
                        width=Self.element_size
                    ](rhs_idx),
                ),
            )

    @staticmethod
    def add[
        SelfLayoutType: TensorLayout,
        self_origin: MutOrigin,
        self_address_space: AddressSpace,
        OtherLayoutType: TensorLayout,
        other_mut: Bool,
        other_origin: Origin[mut=other_mut],
        other_address_space: AddressSpace,
        //,
        dtype: DType,
    ](
        storage: Tuple[
            Self.StorageType[dtype, self_origin, self_address_space],
            SelfLayoutType,
        ],
        other: Tuple[
            Self.StorageType[dtype, other_origin, other_address_space],
            OtherLayoutType,
        ],
    ):
        """Adds `other` into `storage` elementwise, in place.

        Parameters:
            SelfLayoutType: The layout type of the destination storage.
            self_origin: The origin of the destination storage.
            self_address_space: The address space of the destination storage.
            OtherLayoutType: The layout type of the right-hand storage operand.
            other_mut: The mutability of the right-hand storage operand.
            other_origin: The origin of the right-hand storage operand.
            other_address_space: The address space of the right-hand storage
                operand.
            dtype: The element data type of both storages.

        Args:
            storage: A tuple of the destination storage (modified in place) and
                its layout.
            other: A tuple of the right-hand storage operand and its layout.
        """

        @always_inline
        def add(
            lhs: SIMD[dtype, Self.element_size], rhs: type_of(lhs)
        ) -> type_of(lhs):
            return lhs + rhs

        Self._elementwise_binary_with_broadcast(storage, other, add)

    @staticmethod
    def mul[
        SelfLayoutType: TensorLayout,
        self_origin: MutOrigin,
        self_address_space: AddressSpace,
        OtherLayoutType: TensorLayout,
        other_mut: Bool,
        other_origin: Origin[mut=other_mut],
        other_address_space: AddressSpace,
        //,
        dtype: DType,
    ](
        storage: Tuple[
            Self.StorageType[dtype, self_origin, self_address_space],
            SelfLayoutType,
        ],
        other: Tuple[
            Self.StorageType[dtype, other_origin, other_address_space],
            OtherLayoutType,
        ],
    ):
        """Multiplies `storage` by `other` elementwise, in place.

        Parameters:
            SelfLayoutType: The layout type of the destination storage.
            self_origin: The origin of the destination storage.
            self_address_space: The address space of the destination storage.
            OtherLayoutType: The layout type of the right-hand storage operand.
            other_mut: The mutability of the right-hand storage operand.
            other_origin: The origin of the right-hand storage operand.
            other_address_space: The address space of the right-hand storage
                operand.
            dtype: The element data type of both storages.

        Args:
            storage: A tuple of the destination storage (modified in place) and
                its layout.
            other: A tuple of the right-hand storage operand and its layout.
        """

        @always_inline
        def mul(
            lhs: SIMD[dtype, Self.element_size], rhs: type_of(lhs)
        ) -> type_of(lhs):
            return lhs * rhs

        Self._elementwise_binary_with_broadcast(storage, other, mul)

    @staticmethod
    def sub[
        SelfLayoutType: TensorLayout,
        self_origin: MutOrigin,
        self_address_space: AddressSpace,
        OtherLayoutType: TensorLayout,
        other_mut: Bool,
        other_origin: Origin[mut=other_mut],
        other_address_space: AddressSpace,
        //,
        dtype: DType,
    ](
        storage: Tuple[
            Self.StorageType[dtype, self_origin, self_address_space],
            SelfLayoutType,
        ],
        other: Tuple[
            Self.StorageType[dtype, other_origin, other_address_space],
            OtherLayoutType,
        ],
    ):
        """Subtracts `other` from `storage` elementwise, in place.

        Parameters:
            SelfLayoutType: The layout type of the destination storage.
            self_origin: The origin of the destination storage.
            self_address_space: The address space of the destination storage.
            OtherLayoutType: The layout type of the right-hand storage operand.
            other_mut: The mutability of the right-hand storage operand.
            other_origin: The origin of the right-hand storage operand.
            other_address_space: The address space of the right-hand storage
                operand.
            dtype: The element data type of both storages.

        Args:
            storage: A tuple of the destination storage (modified in place) and
                its layout.
            other: A tuple of the right-hand storage operand and its layout.
        """

        @always_inline
        def sub(
            lhs: SIMD[dtype, Self.element_size], rhs: type_of(lhs)
        ) -> type_of(lhs):
            return lhs - rhs

        Self._elementwise_binary_with_broadcast(storage, other, sub)

    @staticmethod
    def floordiv[
        SelfLayoutType: TensorLayout,
        self_origin: MutOrigin,
        self_address_space: AddressSpace,
        OtherLayoutType: TensorLayout,
        other_mut: Bool,
        other_origin: Origin[mut=other_mut],
        other_address_space: AddressSpace,
        //,
        dtype: DType,
    ](
        storage: Tuple[
            Self.StorageType[dtype, self_origin, self_address_space],
            SelfLayoutType,
        ],
        other: Tuple[
            Self.StorageType[dtype, other_origin, other_address_space],
            OtherLayoutType,
        ],
    ):
        """Floor-divides `storage` by `other` elementwise, in place.

        Parameters:
            SelfLayoutType: The layout type of the destination storage.
            self_origin: The origin of the destination storage.
            self_address_space: The address space of the destination storage.
            OtherLayoutType: The layout type of the right-hand storage operand.
            other_mut: The mutability of the right-hand storage operand.
            other_origin: The origin of the right-hand storage operand.
            other_address_space: The address space of the right-hand storage
                operand.
            dtype: The element data type of both storages.

        Args:
            storage: A tuple of the destination storage (modified in place) and
                its layout.
            other: A tuple of the right-hand storage operand and its layout.
        """

        @always_inline
        def floordiv(
            lhs: SIMD[dtype, Self.element_size], rhs: type_of(lhs)
        ) -> type_of(lhs):
            return lhs // rhs

        Self._elementwise_binary_with_broadcast(storage, other, floordiv)

    @staticmethod
    def truediv[
        SelfLayoutType: TensorLayout,
        self_origin: MutOrigin,
        self_address_space: AddressSpace,
        OtherLayoutType: TensorLayout,
        other_mut: Bool,
        other_origin: Origin[mut=other_mut],
        other_address_space: AddressSpace,
        //,
        dtype: DType,
    ](
        storage: Tuple[
            Self.StorageType[dtype, self_origin, self_address_space],
            SelfLayoutType,
        ],
        other: Tuple[
            Self.StorageType[dtype, other_origin, other_address_space],
            OtherLayoutType,
        ],
    ):
        """True-divides `storage` by `other` elementwise, in place.

        Parameters:
            SelfLayoutType: The layout type of the destination storage.
            self_origin: The origin of the destination storage.
            self_address_space: The address space of the destination storage.
            OtherLayoutType: The layout type of the right-hand storage operand.
            other_mut: The mutability of the right-hand storage operand.
            other_origin: The origin of the right-hand storage operand.
            other_address_space: The address space of the right-hand storage
                operand.
            dtype: The element data type of both storages.

        Args:
            storage: A tuple of the destination storage (modified in place) and
                its layout.
            other: A tuple of the right-hand storage operand and its layout.
        """

        @always_inline
        def truediv(
            lhs: SIMD[dtype, Self.element_size], rhs: type_of(lhs)
        ) -> type_of(lhs):
            return lhs / rhs

        Self._elementwise_binary_with_broadcast(storage, other, truediv)


@always_inline
def _device_leaf_ptr[
    dtype: DType, //
](storage: DevicePointer[dtype, _]) -> UnsafePointer[
    Scalar[dtype], MutAnyOrigin, address_space=AddressSpace.GLOBAL
]:
    """Returns the encoded device-leaf pointer held in `storage`'s first bytes.

    A `DevicePointer` encodes to a bare `UnsafePointer` at the kernel boundary
    (see `DevicePointer.device_type`), written into the first bytes of the
    handle's storage slot. On device those bytes are a real device address, so
    this reinterprets them. Aborts on host, where a `DevicePointer` cannot in
    general be dereferenced and its leading bytes are a host reference to the
    owning `DeviceBuffer`.

    The leaf is typed in the `GLOBAL` (device) address space, not `GENERIC`.
    That distinction is invisible on flat-address-space targets (CUDA/HIP),
    where `GENERIC` aliases global memory, but it is load-bearing on targets
    with disjoint address spaces such as Metal: there `GENERIC` is the default
    space (not device memory), and the Metal AIR address-space pass only
    promotes kernel-argument pointers and ptr-bearing *aggregate* blob reloads
    to the device space — not the *scalar* pointer reinterpreted out of the
    handle here. A `GENERIC` leaf would therefore stay in the default space and
    a load/store through it would silently miss device memory.

    Parameters:
        dtype: The element data type of the referenced storage.

    Args:
        storage: The device-pointer handle to reinterpret.

    Returns:
        A bare `UnsafePointer`, in the `GLOBAL` address space, to the
        referenced device storage.
    """
    comptime if is_gpu():
        # Reinterpret the handle's first bytes as the encoded device address.
        # The leaf must be `GLOBAL` (device), not `GENERIC` — see the docstring:
        # a `GENERIC` leaf silently misses device memory on Metal.
        return UnsafePointer(to=storage).bitcast[
            UnsafePointer[
                Scalar[dtype], MutAnyOrigin, address_space=AddressSpace.GLOBAL
            ]
        ]()[]
    else:
        abort("DevicePointerStorage operations are not supported on host")


struct DevicePointerStorage[*, element_width: Int = 1](TensorArith):
    """Implements `TensorArith` backed by a `DevicePointer` handle.

    `DevicePointerStorage` is the device-pointer-backed analogue of
    `PointerStorage`, accepting the same `element_width` parameter. Its
    `StorageType` handle is a `DevicePointer`, which on the host carries the
    buffer's owning reference plus an element offset and size, and which
    substitutes to a bare device `UnsafePointer` at the kernel boundary
    (`DevicePointer.device_type`).

    Because the handle conforms to `DevicePassable`, a host-side
    `DevicePointer` shrinks to a real device address when the enclosing
    `TileTensor` encodes its fields for a kernel launch. The address is written
    into the first bytes of the handle's slot, so the memory operations here
    reinterpret those bytes (`_device_leaf_ptr`) on device. They abort on host:
    device memory is not guaranteed to be host-dereferenceable. The operations
    that don't dereference storage — `offset`, `distance`, `unsafe_cast` — work
    on both host and device using `DevicePointer` arithmetic or pure
    reinterprets.

    Parameters:
        element_width: Number of scalar elements per logical element. A value
            of `1` (the default) is a non-vectorized tensor; larger values
            describe a vectorized view whose logical elements are SIMD vectors.
            The `DevicePointer` handle is always scalar-typed and every
            operation works in scalar-element units, so `element_width` only
            sets `element_size` (and thus the tile's vectorized `ElementType`),
            exactly as for `PointerStorage`.
    """

    comptime element_size = Self.element_width
    """Number of scalar elements per logical element (alias of `element_width`)."""

    comptime StorageType[
        mut: Bool,
        //,
        dtype: DType,
        origin: Origin[mut=mut],
        address_space: AddressSpace,
    ]: DevicePassable & TrivialRegisterPassable = DevicePointer[dtype, origin]
    """A `DevicePointer` handle borrowing the storage.

    The `address_space` is part of the `TensorStorage` interface but is unused:
    a `DevicePointer` always refers to GENERIC device memory.

    Parameters:
        mut: The mutability of the borrowed storage, inferred from `origin`.
        dtype: The element data type of the borrowed storage.
        origin: The origin tracking the lifetime of the borrowed storage.
        address_space: The address space the borrowed storage resides in.
    """

    @staticmethod
    def write_type_name_to(mut writer: Some[Writer]):
        """Write the storage type name representation to the writer.

        Args:
            writer: The `Writer` to output to.
        """
        t"DevicePointerStorage[element_size={Self.element_size}]".write_to(
            writer
        )

    @doc_hidden
    @staticmethod
    @always_inline
    def unsafe_ptr[
        mut: Bool,
        dtype: DType,
        origin: Origin[mut=mut],
        address_space: AddressSpace,
        //,
    ](
        storage: Self.StorageType[dtype, origin, address_space],
    ) -> UnsafePointer[Scalar[dtype], origin, address_space=address_space]:
        """Returns a raw scalar pointer to the base of the borrowed storage.

        On device the owning `DeviceBuffer` is unavailable, so this reinterprets
        the encoded device address out of the handle (`_device_leaf_ptr`); on
        host it recovers the raw (offset-adjusted) address through the owning
        `DeviceBuffer`. Stopgap for the in-progress `LayoutTensor` migration,
        see GPUA-6.

        Parameters:
            mut: The mutability of the borrowed storage, inferred from `origin`.
            dtype: The element data type of the borrowed storage.
            origin: The origin tracking the lifetime of the borrowed storage.
            address_space: The address space the borrowed storage resides in.

        Args:
            storage: The storage to recover the base pointer from.

        Returns:
            A bare `UnsafePointer` to the first scalar element of the storage.
        """
        comptime ResultPtr = UnsafePointer[
            Scalar[dtype], origin, address_space=address_space
        ]
        # `_device_leaf_ptr` returns a `GLOBAL` (device) leaf because Metal has
        # no usable `GENERIC` device space and the compiler does not insert the
        # address-space conversion itself. `rebind` cannot change a pointer's
        # address space, so cast the leaf to the tile's declared space, then
        # `rebind` the origin. On flat-address targets (CUDA/HIP) `GLOBAL` and
        # `GENERIC` are the same address so this cast is a no-op reinterpret; on
        # segmented targets like Metal a `GENERIC` result cannot reach device
        # memory, but `.ptr`'s return type is fixed to the tile's (`GENERIC`)
        # space, so Metal is out of scope for this stopgap (see GPUA-6).
        comptime if is_gpu():
            return rebind[ResultPtr](
                _device_leaf_ptr(storage).address_space_cast[address_space]()
            )
        else:
            return rebind[ResultPtr](
                storage.unsafe_ptr().address_space_cast[address_space]()
            )

    @staticmethod
    @always_inline
    def unsafe_cast[
        to_mut: Bool,
        //,
        to_dtype: DType,
        to_origin: Origin[mut=to_mut],
        to_address_space: AddressSpace,
    ](
        storage: Self.StorageType[...],
        out result: Self.StorageType[
            mut=to_mut, to_dtype, to_origin, to_address_space
        ],
    ):
        """Reinterprets a storage handle with new type parameters.

        `DevicePointer` has an identical layout across `dtype` and `origin`, so
        this is a byte-for-byte reinterpret of the handle; no `DeviceBuffer`
        element conversion takes place. The caller is responsible for ensuring
        the new parameters are valid for the referenced storage.

        Parameters:
            to_mut: The mutability of the origin.
            to_dtype: The element data type to reinterpret the storage as.
            to_origin: The origin to reinterpret the storage as.
            to_address_space: The address space to reinterpret the storage as.

        Args:
            storage: The storage to reinterpret.

        Returns:
            A handle referring to the same storage, viewed with the new type
            parameters.
        """
        result = UnsafePointer(to=storage).bitcast[type_of(result)]()[]

    @staticmethod
    @always_inline
    def load[
        dtype: DType,
        //,
        width: SIMDSize,
        alignment: Int,
        invariant: Bool = False,
        non_temporal: Bool = False,
    ](storage: Self.StorageType[mut=False, dtype, ...]) -> SIMD[dtype, width]:
        """Loads a `SIMD` value from the storage.

        Device-only: reinterprets the encoded device pointer, which aborts on
        host. Device memory is not guaranteed to be host-dereferenceable.

        Parameters:
            dtype: The element data type of the storage.
            width: The number of elements to load.
            alignment: The alignment guarantee for the load.
            invariant: If True, the compiler may assume the memory won't be
                modified during the kernel, enabling load hoisting and caching.
            non_temporal: If True, indicates the data will not be reused soon,
                allowing the hardware to bypass caches (e.g., streaming loads).

        Args:
            storage: The storage to load from.

        Returns:
            The loaded `SIMD` value.
        """
        return _device_leaf_ptr(storage).load[
            width=width,
            alignment=alignment,
            invariant=invariant,
            non_temporal=non_temporal,
        ]()

    @staticmethod
    @always_inline
    def load[
        dtype: DType,
        //,
        width: SIMDSize,
        alignment: Int,
        invariant: Bool = False,
        non_temporal: Bool = False,
    ](
        storage: Self.StorageType[mut=False, dtype, ...],
        offset: Some[Indexer],
    ) -> SIMD[dtype, width]:
        """Loads a `SIMD` value at a scalar-element offset from the storage.

        Device-only: reinterprets the encoded device pointer, which aborts on
        host. Device memory is not guaranteed to be host-dereferenceable.

        Parameters:
            dtype: The element data type of the storage.
            width: The number of elements to load.
            alignment: The alignment guarantee for the load.
            invariant: If True, the compiler may assume the memory won't be
                modified during the kernel, enabling load hoisting and caching.
            non_temporal: If True, indicates the data will not be reused soon,
                allowing the hardware to bypass caches (e.g., streaming loads).

        Args:
            storage: The storage to load from.
            offset: The scalar-element offset to load at.

        Returns:
            The loaded `SIMD` value.
        """
        return _device_leaf_ptr(storage).load[
            width=width,
            alignment=alignment,
            invariant=invariant,
            non_temporal=non_temporal,
        ](offset)

    @staticmethod
    @always_inline
    def store[
        dtype: DType,
        alignment: Int,
        *,
        non_temporal: Bool = False,
    ](storage: Self.StorageType[mut=True, dtype, ...], value: SIMD[dtype, _],):
        """Stores a `SIMD` value into the storage.

        Device-only: reinterprets the encoded device pointer, which aborts on
        host. Device memory is not guaranteed to be host-dereferenceable.

        Parameters:
            dtype: The element data type of the storage.
            alignment: The alignment guarantee for the store.
            non_temporal: If True, indicates the data will not be reused soon,
                allowing the hardware to bypass caches (e.g., streaming stores).

        Args:
            storage: The storage to store into.
            value: The `SIMD` value to store.
        """
        _device_leaf_ptr(storage).store[
            alignment=alignment, non_temporal=non_temporal
        ](value)

    @staticmethod
    @always_inline
    def store[
        dtype: DType,
        alignment: Int,
        *,
        non_temporal: Bool = False,
    ](
        storage: Self.StorageType[mut=True, dtype, ...],
        offset: Some[Indexer],
        value: SIMD[dtype, _],
    ):
        """Stores a `SIMD` value at a scalar-element offset in the storage.

        Device-only: reinterprets the encoded device pointer, which aborts on
        host. Device memory is not guaranteed to be host-dereferenceable.

        Parameters:
            dtype: The element data type of the storage.
            alignment: The alignment guarantee for the store.
            non_temporal: If True, indicates the data will not be reused soon,
                allowing the hardware to bypass caches (e.g., streaming stores).

        Args:
            storage: The storage to store into.
            offset: The scalar-element offset to store at.
            value: The `SIMD` value to store.
        """
        _device_leaf_ptr(storage).store[
            alignment=alignment, non_temporal=non_temporal
        ](offset, value)

    @staticmethod
    @always_inline
    def offset(
        storage: Self.StorageType[...], offset: Some[Indexer]
    ) -> type_of(storage):
        """Returns a storage handle offset by a number of scalar elements.

        On host this advances the wrapped `DevicePointer` (bounds-checked
        against the owning `DeviceBuffer`). On device it advances the encoded
        device pointer held in the handle's first bytes, preserving the rest of
        the handle's (unused) bytes.

        Args:
            storage: The storage to offset from.
            offset: The number of scalar elements to advance the handle by.

        Returns:
            A handle of the same type starting `offset` scalar elements into
            the referenced storage.
        """
        comptime if is_gpu():
            var result = storage
            var leaf = UnsafePointer(to=result).bitcast[
                UnsafePointer[Scalar[type_of(storage).dtype], MutAnyOrigin]
            ]()
            leaf[] = leaf[] + offset
            return result
        else:
            # Keep this non-raising (matching the pointer-backed policy and
            # `TileTensor`'s `DeviceBuffer` constructor) by aborting on the
            # out-of-bounds case `DevicePointer` arithmetic raises on.
            try:
                return storage + index(offset)
            except e:
                abort(String("DevicePointerStorage.offset: ", e))

    @staticmethod
    def distance[
        dtype: DType, address_space: AddressSpace, //
    ](
        storage: Self.StorageType[mut=False, dtype, _, address_space],
        other: Self.StorageType[mut=False, dtype, _, address_space],
    ) -> Int:
        """Returns the scalar-element distance from `other` to `storage`.

        Parameters:
            dtype: The storages' `DType`.
            address_space: The storages' `AddressSpace`.

        Args:
            storage: The storage to measure the distance to.
            other: The storage to measure the distance from.

        Returns:
            The number of scalar elements separating the two handles. The
            value is positive when `storage` is ahead of `other` and negative
            when it precedes `other`.
        """
        comptime if is_gpu():
            return (
                Int(_device_leaf_ptr(storage)) - Int(_device_leaf_ptr(other))
            ) // size_of[dtype]()
        else:
            # The element offsets are available on host without dereferencing;
            # their difference is the scalar-element distance (element_size 1).
            return storage.offset() - other.offset()

    @always_inline
    @staticmethod
    def _elementwise_binary_with_broadcast[
        SelfLayoutType: TensorLayout,
        self_origin: MutOrigin,
        OtherLayoutType: TensorLayout,
        other_mut: Bool,
        other_origin: Origin[mut=other_mut],
        //,
        dtype: DType,
    ](
        storage: Tuple[
            Self.StorageType[dtype, self_origin, AddressSpace.GENERIC],
            SelfLayoutType,
        ],
        other: Tuple[
            Self.StorageType[dtype, other_origin, AddressSpace.GENERIC],
            OtherLayoutType,
        ],
        func: Some[
            def(
                SIMD[dtype, Self.element_width], SIMD[dtype, Self.element_width]
            ) -> (SIMD[dtype, Self.element_width])
        ],
    ):
        """Apply an elementwise binary operation with broadcasting support.

        This internal method applies a binary operation between elements of this
        tensor and another tensor, with support for limited broadcasting
        patterns. The operation is performed in-place on this tensor.

        Parameters:
            SelfLayoutType: The layout type of the destination storage.
            self_origin: The origin of the destination storage.
            OtherLayoutType: The layout type of the right-hand storage operand.
            other_mut: The mutability of the right-hand storage operand.
            other_origin: The origin of the right-hand storage operand.
            dtype: The dtype of both tensors's elements.

        Args:
            storage: A tuple of the destination storage (modified in place) and
                its layout.
            other: A tuple of the right-hand storage operand and its layout.
            func: A binary function that takes two elements (one from each
                tensor) and returns a single element as the result of the
                operation.

        Notes:

        - Currently supports only rank-2 tensors or tensors of the same rank.
        - For tensors of the same rank, shapes must match exactly.
        - For rank-1 to rank-2 broadcasting, the rank-1 tensor's dimension must
            match the corresponding dimension of the rank-2 tensor.
        - The operation is optimized based on the memory layout of both tensors.
        """

        ref self_storage = storage[0]
        ref self_layout = storage[1]
        ref other_storage = other[0]
        ref other_layout = other[1]

        comptime self_rank = type_of(self_layout).rank
        comptime other_rank = type_of(other_layout).rank
        comptime other_shape[i: Int] = type_of(other_layout).static_shape[i]
        comptime self_shape[i: Int] = type_of(self_layout).static_shape[i]

        comptime if self_rank == other_rank:
            comptime for axis in range(type_of(self_layout).rank):
                comptime assert other_shape[axis] == self_shape[axis], (
                    "_elementwise_binary_with_broadcast requires shape to"
                    " be the same for tensors of the same rank"
                )

        comptime assert type_of(self_layout).all_dims_known, (
            "_elementwise_binary_with_broadcast must operates on tensors"
            " of statically know layouts"
        )
        comptime assert other_rank <= self_rank, (
            "_elementwise_binary_with_broadcast must operates on tensor of"
            " equal of lower rank"
        )

        # TODO(KERN-812): Support numpy like broadcasting and relax rank-2
        # constrain.
        comptime assert (
            self_rank == 2 or self_rank == other_rank
        ), "Only supports rank-2 tensor, or same rank"

        comptime if other_rank == 1:
            comptime assert other_shape[0] == self_shape[0], (
                "_elementwise_binary_with_broadcast 1d tensor operand must"
                " have a dim that matches the tensors"
            )

            comptime for i in range(type_of(self_layout).static_product):
                comptime other_size = type_of(other_layout).static_product

                var lhs_idx = self_layout(Idx[i])
                var rhs_idx = other_layout(Idx[i % other_size])

                _device_leaf_ptr(self_storage).store(
                    lhs_idx,
                    func(
                        _device_leaf_ptr(self_storage).load[
                            width=Self.element_width
                        ](lhs_idx),
                        _device_leaf_ptr(other_storage).load[
                            width=Self.element_width
                        ](rhs_idx),
                    ),
                )

        comptime for i in range(type_of(self_layout).static_product):
            var lhs_idx = self_layout(Idx[i])
            var rhs_idx = other_layout(Idx[i])
            _device_leaf_ptr(self_storage).store(
                lhs_idx,
                func(
                    _device_leaf_ptr(self_storage).load[
                        width=Self.element_size
                    ](lhs_idx),
                    _device_leaf_ptr(other_storage).load[
                        width=Self.element_size
                    ](rhs_idx),
                ),
            )

    @staticmethod
    def add[
        SelfLayoutType: TensorLayout,
        self_origin: MutOrigin,
        self_address_space: AddressSpace,
        OtherLayoutType: TensorLayout,
        other_mut: Bool,
        other_origin: Origin[mut=other_mut],
        other_address_space: AddressSpace,
        //,
        dtype: DType,
    ](
        storage: Tuple[
            Self.StorageType[dtype, self_origin, self_address_space],
            SelfLayoutType,
        ],
        other: Tuple[
            Self.StorageType[dtype, other_origin, other_address_space],
            OtherLayoutType,
        ],
    ):
        """Adds `other` into `storage` elementwise, in place.

        Parameters:
            SelfLayoutType: The layout type of the destination storage.
            self_origin: The origin of the destination storage.
            self_address_space: The address space of the destination storage.
            OtherLayoutType: The layout type of the right-hand storage operand.
            other_mut: The mutability of the right-hand storage operand.
            other_origin: The origin of the right-hand storage operand.
            other_address_space: The address space of the right-hand storage
                operand.
            dtype: The element data type of both storages.

        Args:
            storage: A tuple of the destination storage (modified in place) and
                its layout.
            other: A tuple of the right-hand storage operand and its layout.
        """

        @always_inline
        def add(
            lhs: SIMD[dtype, Self.element_size], rhs: type_of(lhs)
        ) -> type_of(lhs):
            return lhs + rhs

        Self._elementwise_binary_with_broadcast(storage, other, add)

    @staticmethod
    def mul[
        SelfLayoutType: TensorLayout,
        self_origin: MutOrigin,
        self_address_space: AddressSpace,
        OtherLayoutType: TensorLayout,
        other_mut: Bool,
        other_origin: Origin[mut=other_mut],
        other_address_space: AddressSpace,
        //,
        dtype: DType,
    ](
        storage: Tuple[
            Self.StorageType[dtype, self_origin, self_address_space],
            SelfLayoutType,
        ],
        other: Tuple[
            Self.StorageType[dtype, other_origin, other_address_space],
            OtherLayoutType,
        ],
    ):
        """Multiplies `storage` by `other` elementwise, in place.

        Parameters:
            SelfLayoutType: The layout type of the destination storage.
            self_origin: The origin of the destination storage.
            self_address_space: The address space of the destination storage.
            OtherLayoutType: The layout type of the right-hand storage operand.
            other_mut: The mutability of the right-hand storage operand.
            other_origin: The origin of the right-hand storage operand.
            other_address_space: The address space of the right-hand storage
                operand.
            dtype: The element data type of both storages.

        Args:
            storage: A tuple of the destination storage (modified in place) and
                its layout.
            other: A tuple of the right-hand storage operand and its layout.
        """

        @always_inline
        def mul(
            lhs: SIMD[dtype, Self.element_size], rhs: type_of(lhs)
        ) -> type_of(lhs):
            return lhs * rhs

        Self._elementwise_binary_with_broadcast(storage, other, mul)

    @staticmethod
    def sub[
        SelfLayoutType: TensorLayout,
        self_origin: MutOrigin,
        self_address_space: AddressSpace,
        OtherLayoutType: TensorLayout,
        other_mut: Bool,
        other_origin: Origin[mut=other_mut],
        other_address_space: AddressSpace,
        //,
        dtype: DType,
    ](
        storage: Tuple[
            Self.StorageType[dtype, self_origin, self_address_space],
            SelfLayoutType,
        ],
        other: Tuple[
            Self.StorageType[dtype, other_origin, other_address_space],
            OtherLayoutType,
        ],
    ):
        """Subtracts `other` from `storage` elementwise, in place.

        Parameters:
            SelfLayoutType: The layout type of the destination storage.
            self_origin: The origin of the destination storage.
            self_address_space: The address space of the destination storage.
            OtherLayoutType: The layout type of the right-hand storage operand.
            other_mut: The mutability of the right-hand storage operand.
            other_origin: The origin of the right-hand storage operand.
            other_address_space: The address space of the right-hand storage
                operand.
            dtype: The element data type of both storages.

        Args:
            storage: A tuple of the destination storage (modified in place) and
                its layout.
            other: A tuple of the right-hand storage operand and its layout.
        """

        @always_inline
        def sub(
            lhs: SIMD[dtype, Self.element_size], rhs: type_of(lhs)
        ) -> type_of(lhs):
            return lhs - rhs

        Self._elementwise_binary_with_broadcast(storage, other, sub)

    @staticmethod
    def floordiv[
        SelfLayoutType: TensorLayout,
        self_origin: MutOrigin,
        self_address_space: AddressSpace,
        OtherLayoutType: TensorLayout,
        other_mut: Bool,
        other_origin: Origin[mut=other_mut],
        other_address_space: AddressSpace,
        //,
        dtype: DType,
    ](
        storage: Tuple[
            Self.StorageType[dtype, self_origin, self_address_space],
            SelfLayoutType,
        ],
        other: Tuple[
            Self.StorageType[dtype, other_origin, other_address_space],
            OtherLayoutType,
        ],
    ):
        """Floor-divides `storage` by `other` elementwise, in place.

        Parameters:
            SelfLayoutType: The layout type of the destination storage.
            self_origin: The origin of the destination storage.
            self_address_space: The address space of the destination storage.
            OtherLayoutType: The layout type of the right-hand storage operand.
            other_mut: The mutability of the right-hand storage operand.
            other_origin: The origin of the right-hand storage operand.
            other_address_space: The address space of the right-hand storage
                operand.
            dtype: The element data type of both storages.

        Args:
            storage: A tuple of the destination storage (modified in place) and
                its layout.
            other: A tuple of the right-hand storage operand and its layout.
        """

        @always_inline
        def floordiv(
            lhs: SIMD[dtype, Self.element_size], rhs: type_of(lhs)
        ) -> type_of(lhs):
            return lhs // rhs

        Self._elementwise_binary_with_broadcast(storage, other, floordiv)

    @staticmethod
    def truediv[
        SelfLayoutType: TensorLayout,
        self_origin: MutOrigin,
        self_address_space: AddressSpace,
        OtherLayoutType: TensorLayout,
        other_mut: Bool,
        other_origin: Origin[mut=other_mut],
        other_address_space: AddressSpace,
        //,
        dtype: DType,
    ](
        storage: Tuple[
            Self.StorageType[dtype, self_origin, self_address_space],
            SelfLayoutType,
        ],
        other: Tuple[
            Self.StorageType[dtype, other_origin, other_address_space],
            OtherLayoutType,
        ],
    ):
        """True-divides `storage` by `other` elementwise, in place.

        Parameters:
            SelfLayoutType: The layout type of the destination storage.
            self_origin: The origin of the destination storage.
            self_address_space: The address space of the destination storage.
            OtherLayoutType: The layout type of the right-hand storage operand.
            other_mut: The mutability of the right-hand storage operand.
            other_origin: The origin of the right-hand storage operand.
            other_address_space: The address space of the right-hand storage
                operand.
            dtype: The element data type of both storages.

        Args:
            storage: A tuple of the destination storage (modified in place) and
                its layout.
            other: A tuple of the right-hand storage operand and its layout.
        """

        @always_inline
        def truediv(
            lhs: SIMD[dtype, Self.element_size], rhs: type_of(lhs)
        ) -> type_of(lhs):
            return lhs / rhs

        Self._elementwise_binary_with_broadcast(storage, other, truediv)
