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
"""Allocation-handle-backed storage for tile tensors.

`TensorBufferStorage` lets a `TileTensor` be built over a HAL `Buffer` without a
host-visible device address. The `TensorBufferStorageView` handle carries the
allocation handle plus a byte range, so it works on backends (e.g. NPUs)
where an allocation has no address until a launch maps it; the launch
boundary is what resolves the handle, never host code.
"""

from std.os import abort
from std.sys import size_of

from layout import Coord, CoordLike, TensorLayout
from layout.tensor_storage import TensorStorage
from layout.tile_tensor import TileTensor

from max.gpu.host._hal.buffer import Buffer, BufferView
from max.gpu.host._hal.device import DeviceSpec
from max.gpu.host._hal.plugin import MemoryHandle


@fieldwise_init
struct TensorBufferStorageView[
    mut: Bool,
    //,
    dtype: DType,
    origin: Origin[mut=mut],
    address_space: AddressSpace,
](TrivialRegisterPassable):
    """A typed, non-owning view over a range of a HAL device allocation.

    This is `TensorBufferStorage`'s storage handle: an allocation handle plus a byte
    range, never a device address. It cannot be dereferenced on the host; a
    kernel launch resolves the handle to device memory when the launch is
    encoded. Create one through the owning `TensorBufferStorage` rather than
    directly.

    Parameters:
        mut: The mutability of the viewed storage, inferred from `origin`.
        dtype: The element data type of the viewed storage.
        origin: The origin tracking the lifetime of the viewed storage.
        address_space: The address space the storage resides in. Carried for
            the `TensorStorage` interface; device allocations are GENERIC.
    """

    var _memory: MemoryHandle
    var _byte_offset: UInt64
    var _byte_size: UInt64

    def hal_view(self) -> BufferView:
        """Returns the untyped HAL `BufferView` over the same range.

        Returns:
            A `BufferView` suitable for HAL copy and fill APIs.
        """
        return BufferView(self._memory, self._byte_offset, self._byte_size)


struct TensorBufferStorage[
    buffer_dtype: DType,
    device_specification: DeviceSpec,
](Movable, TensorStorage):
    """A typed wrapper over a HAL `Buffer`, usable as `TileTensor` storage.

    Implements `TensorStorage` with a `TensorBufferStorageView` handle, so a tensor
    built over it carries the allocation handle — not a device address — to
    the kernel launch boundary. Host code can slice and describe the tensor
    (`offset`, `distance`, views) but never dereference it: device memory is
    not host-addressable on every backend this abstraction targets.

    Views handed out use `MutUntrackedOrigin`; the wrapped `Buffer` owns the
    allocation and must outlive them.

    Parameters:
        buffer_dtype: The element data type stored in the buffer.
        device_specification: The compilation target whose memory the wrapped
            buffer lives on.
    """

    comptime _BASE_TYPE_NAME: StaticString = "TensorBufferStorage"
    """The unparameterized name of this storage policy."""

    comptime StorageType[
        mut: Bool,
        //,
        dtype: DType,
        origin: Origin[mut=mut],
        address_space: AddressSpace,
    ]: TrivialRegisterPassable = TensorBufferStorageView[
        dtype, origin, address_space
    ]
    """A `TensorBufferStorageView` handle over the borrowed storage.

    Parameters:
        mut: The mutability of the borrowed storage, inferred from `origin`.
        dtype: The element data type of the borrowed storage.
        origin: The origin tracking the lifetime of the borrowed storage.
        address_space: The address space the storage resides in.
    """

    var _buffer: Buffer[Self.device_specification]

    def __init__(out self, var buffer: Buffer[Self.device_specification]):
        """Wraps `buffer` as typed tensor storage.

        Args:
            buffer: The allocation to wrap. Its byte size must be a multiple
                of the element size.
        """
        debug_assert(
            buffer.byte_size % UInt64(size_of[Self.buffer_dtype]()) == 0,
            "buffer byte size must be a multiple of the element size",
        )
        self._buffer = buffer^

    def buffer(ref self) -> ref[self._buffer] Buffer[Self.device_specification]:
        """Returns a reference to the wrapped `Buffer`.

        Returns:
            A reference to the wrapped allocation.
        """
        return self._buffer

    def unwrap(deinit self) -> Buffer[Self.device_specification]:
        """Consumes the wrapper and returns the wrapped `Buffer`.

        Returns:
            The wrapped allocation, e.g. to pass to `Context.free_sync`.
        """
        return self._buffer^

    def view(
        self,
    ) -> TensorBufferStorageView[
        Self.buffer_dtype, MutUntrackedOrigin, AddressSpace.GENERIC
    ]:
        """Returns a storage handle over the whole allocation.

        Returns:
            A view spanning every element of the buffer.
        """
        return {self._buffer._handle, 0, self._buffer.byte_size}

    def view(
        self, *, element_offset: Int, element_count: Int
    ) -> TensorBufferStorageView[
        Self.buffer_dtype, MutUntrackedOrigin, AddressSpace.GENERIC
    ]:
        """Returns a storage handle over a sub-range of the allocation.

        Args:
            element_offset: The first element of the range.
            element_count: The number of elements in the range.

        Returns:
            A view spanning the requested elements.
        """
        comptime elem_size = size_of[Self.buffer_dtype]()
        var byte_offset = UInt64(element_offset * elem_size)
        var byte_size = UInt64(element_count * elem_size)
        debug_assert(
            byte_offset + byte_size <= self._buffer.byte_size,
            "TensorBufferStorage view range exceeds the allocation",
        )
        return {self._buffer._handle, byte_offset, byte_size}

    def tile_tensor[
        LayoutType: TensorLayout
    ](self, var layout: LayoutType) -> TileTensor[
        Self.buffer_dtype, LayoutType, MutUntrackedOrigin, Storage=Self
    ]:
        """Returns a `TileTensor` over the whole allocation with `layout`.

        Parameters:
            LayoutType: The layout type describing the tensor's shape.

        Args:
            layout: The layout of the returned tensor. Its element count must
                not exceed the buffer's.

        Returns:
            A tensor whose storage is a handle into this buffer.
        """
        return {self.view(), layout}

    # ===------------------------------------------------------------------=== #
    # TensorStorage interface
    # ===------------------------------------------------------------------=== #

    @staticmethod
    def write_type_name_to(mut writer: Some[Writer]):
        """Writes the storage type name representation to the writer.

        Args:
            writer: The `Writer` to output to.
        """
        t"TensorBufferStorage[dtype={Self.buffer_dtype}]".write_to(writer)

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
        """Raises: buffer-handle storage has no host-visible address.

        Parameters:
            mut: The mutability of the borrowed storage, inferred from `origin`.
            dtype: The element data type of the borrowed storage.
            origin: The origin tracking the lifetime of the borrowed storage.
            address_space: The address space the storage resides in.

        Args:
            storage: The storage a pointer was requested for.

        Returns:
            Never returns.

        Raises:
            Always: the allocation has no address until a launch maps it.
        """
        raise Error(
            "TensorBufferStorage storage is an allocation handle with no"
            " host-visible device address"
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

        The handle's byte range is preserved; no element conversion takes
        place. The caller is responsible for ensuring the new parameters are
        valid for the referenced storage.

        Parameters:
            to_mut: The mutability of the origin.
            to_dtype: The element data type to reinterpret the storage as.
            to_origin: The origin to reinterpret the storage as.
            to_address_space: The address space to reinterpret the storage as.

        Args:
            storage: The storage to reinterpret.

        Returns:
            A handle over the same byte range with the new type parameters.
        """
        result = {storage._memory, storage._byte_offset, storage._byte_size}

    @staticmethod
    def load[
        dtype: DType,
        //,
        width: SIMDLength,
        alignment: Int,
        invariant: Bool = False,
        non_temporal: Bool = False,
    ](storage: Self.StorageType[mut=False, dtype, ...]) -> SIMD[dtype, width]:
        """Aborts: buffer-handle storage cannot be dereferenced on the host.

        Parameters:
            dtype: The element data type of the storage.
            width: The number of elements to load.
            alignment: The alignment guarantee for the load.
            invariant: Unused; part of the `TensorStorage` interface.
            non_temporal: Unused; part of the `TensorStorage` interface.

        Args:
            storage: The storage to load from.

        Returns:
            Never returns.
        """
        abort("TensorBufferStorage storage cannot be dereferenced on the host")

    @staticmethod
    def load[
        dtype: DType,
        //,
        width: SIMDLength,
        alignment: Int,
        invariant: Bool = False,
        non_temporal: Bool = False,
    ](
        storage: Self.StorageType[mut=False, dtype, ...],
        offset: Some[Indexer],
    ) -> SIMD[dtype, width]:
        """Aborts: buffer-handle storage cannot be dereferenced on the host.

        Parameters:
            dtype: The element data type of the storage.
            width: The number of elements to load.
            alignment: The alignment guarantee for the load.
            invariant: Unused; part of the `TensorStorage` interface.
            non_temporal: Unused; part of the `TensorStorage` interface.

        Args:
            storage: The storage to load from.
            offset: The scalar-element offset to load at.

        Returns:
            Never returns.
        """
        abort("TensorBufferStorage storage cannot be dereferenced on the host")

    @staticmethod
    def store[
        dtype: DType,
        alignment: Int,
        *,
        non_temporal: Bool = False,
    ](storage: Self.StorageType[mut=True, dtype, ...], value: SIMD[dtype, _]):
        """Aborts: buffer-handle storage cannot be dereferenced on the host.

        Parameters:
            dtype: The element data type of the storage.
            alignment: The alignment guarantee for the store.
            non_temporal: Unused; part of the `TensorStorage` interface.

        Args:
            storage: The storage to store into.
            value: The `SIMD` value that would be stored.
        """
        abort("TensorBufferStorage storage cannot be dereferenced on the host")

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
        """Aborts: buffer-handle storage cannot be dereferenced on the host.

        Parameters:
            dtype: The element data type of the storage.
            alignment: The alignment guarantee for the store.
            non_temporal: Unused; part of the `TensorStorage` interface.

        Args:
            storage: The storage to store into.
            offset: The scalar-element offset that would be stored at.
            value: The `SIMD` value that would be stored.
        """
        abort("TensorBufferStorage storage cannot be dereferenced on the host")

    comptime OffsetResultType[
        offset_types: TypeList[Trait=CoordLike, ...],
    ]: TensorStorage = Self
    """The storage type produced by offsetting with a given coordinate.

    Offsetting never changes the storage policy, so this is `Self`.

    Parameters:
        offset_types: The coordinate element types of the applied offset.
    """

    @staticmethod
    @always_inline
    def offset[
        offset_mut: Bool,
        offset_types: TypeList[Trait=CoordLike, ...],
        //,
        offset_dtype: DType,
        offset_origin: Origin[mut=offset_mut],
        offset_address_space: AddressSpace,
    ](
        var storage: Self.StorageType[
            offset_dtype, offset_origin, offset_address_space
        ],
        var offset_coord: Coord[*offset_types],
    ) -> Self.OffsetResultType[offset_types].StorageType[
        offset_dtype, offset_origin, offset_address_space
    ]:
        """Returns a storage handle offset by a number of scalar elements.

        Advances the handle's byte range without touching memory: the range's
        start moves forward and its size shrinks by the same amount, so the
        handle always describes the remainder of the original view.

        Parameters:
            offset_mut: The mutability of the storage, inferred from
                `offset_origin`.
            offset_types: The coordinate element types of `offset_coord`.
            offset_dtype: The element data type of the storage.
            offset_origin: The origin tracking the lifetime of the storage.
            offset_address_space: The address space the storage resides in.

        Args:
            storage: The storage to offset from.
            offset_coord: A rank-1 coordinate holding the number of scalar
                elements to advance the handle by.

        Returns:
            A handle of the same type starting the given number of scalar
            elements into the referenced storage.
        """
        comptime assert offset_coord.flat_rank == 1
        var delta = Int(offset_coord[0].value()) * size_of[offset_dtype]()
        debug_assert(
            -delta <= Int(storage._byte_offset)
            and delta <= Int(storage._byte_size),
            "TensorBufferStorage offset is outside the view's byte range",
        )
        storage._byte_offset = UInt64(Int(storage._byte_offset) + delta)
        storage._byte_size = UInt64(Int(storage._byte_size) - delta)
        return storage

    @staticmethod
    def distance[
        dtype: DType, address_space: AddressSpace, //
    ](
        storage: Self.StorageType[mut=False, dtype, _, address_space],
        other: Self.StorageType[mut=False, dtype, _, address_space],
    ) -> Int:
        """Returns the scalar-element distance from `other` to `storage`.

        Both handles must view the same allocation; the distance is the
        difference of their byte offsets, which is available on the host
        without dereferencing.

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
        return (Int(storage._byte_offset) - Int(other._byte_offset)) // size_of[
            dtype
        ]()

    @staticmethod
    def copy_from[
        SelfLayoutType: TensorLayout,
        self_origin: MutOrigin,
        self_address_space: AddressSpace,
        OtherLayoutType: TensorLayout,
        other_mut: Bool,
        other_origin: Origin[mut=other_mut],
        other_address_space: AddressSpace,
        //,
        dst_dtype: DType,
        src_dtype: DType,
        OtherStorage: TensorStorage,
    ](
        storage: Tuple[
            Self.StorageType[dst_dtype, self_origin, self_address_space],
            SelfLayoutType,
        ],
        other: Tuple[
            OtherStorage.StorageType[
                src_dtype, other_origin, other_address_space
            ],
            OtherLayoutType,
        ],
    ):
        """Aborts: buffer-handle storage cannot be dereferenced on the host.

        Use the HAL copy APIs (via `TensorBufferStorageView.hal_view()`) to move data
        between host and device instead.

        Parameters:
            SelfLayoutType: The layout type of the destination storage.
            self_origin: The origin of the destination storage.
            self_address_space: The address space of the destination storage.
            OtherLayoutType: The layout type of the source storage.
            other_mut: The mutability of the source storage.
            other_origin: The origin of the source storage.
            other_address_space: The address space of the source storage.
            dst_dtype: The element data type of the destination storage.
            src_dtype: The element data type of the source storage.
            OtherStorage: The storage policy of the source.

        Args:
            storage: A tuple of the destination storage and its layout.
            other: A tuple of the source storage and its layout.
        """
        abort("TensorBufferStorage storage cannot be dereferenced on the host")
