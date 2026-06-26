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


from std.sys import size_of


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


struct PointerStorage[*, element_width: Int = 1](TensorStorage):
    """Implements `TensorStorage` backed by a raw `UnsafePointer`.

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
        return __mlir_op.`pop.pointer.bitcast`[
            _type=type_of(result)._mlir_type,
        ](storage.address)

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
