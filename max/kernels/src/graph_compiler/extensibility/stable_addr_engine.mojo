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
"""Defines the tensor engine backed by a device graph stable address slot."""

from std.os import abort
from std.sys.info import is_gpu
from max.gpu.host import StableAddr
from layout import Coord, CoordLike, TensorLayout, TileTensor
from layout.tensor_engine import TensorEngine, _copy_from, _offset_elements


@fieldwise_init
struct StableStorage[
    mut: Bool,
    //,
    dtype: DType,
    origin: Origin[mut=mut],
    address_space: AddressSpace,
](TrivialRegisterPassable):
    """The storage handle of `StableAddrEngine`: a stable address slot plus a
    scalar-element offset into the tensor the slot currently stands for.

    The handle never holds the data pointer itself. Every dereference reads
    the pointer out of the slot first, so a view built from the handle (a
    tile, a slice, an offset) keeps going through the slot no matter where it
    was built. That is what lets a `TileTensor` over a slot be constructed and
    tiled on the host while a graph recorded against it still sees whatever
    `DeviceGraphBuilder.place_stable()` puts in the slot at replay.

    The type parameters are carried for `TensorEngine` conformance only. The
    slot and its contents are always `GENERIC` device addresses, and the
    `origin` is not tracked by the slot.

    Parameters:
        mut: The mutability of the viewed storage, inferred from `origin`.
        dtype: The element data type of the viewed storage.
        origin: The origin the view is attributed to.
        address_space: The address space of the viewed storage.
    """

    var addr: StableAddr
    """The slot whose contents locate the tensor data."""

    var offset: Int64
    """Scalar elements between the slot's data pointer and this handle."""

    @inline(.always)
    def __init__(out self, addr: StableAddr):
        """Initialize the stable storage from a stable address with 0 offset."""
        self = Self(addr, 0)

    @inline(.always)
    def __init__(out self, ptr: Pointer[NoneType, _]):
        """Initializes the stable storage over the slot at address `ptr`, with
        0 offset.

        `ptr` is reinterpreted as the slot's address and never read through.
        The graph compiler hands a stable argument's slot where a dense
        argument's data pointer would go, and this constructor is where the
        two meet.

        Args:
            ptr: The slot's address.
        """
        self = Self(
            StableAddr(
                ptr.unsafe_bitcast[Pointer[NoneType, MutUntrackedOrigin]]()
                .as_imm()
                .unsafe_origin_cast[ImmUntrackedOrigin]()
            ),
            0,
        )

    @inline(.always)
    def _leaf_ptr(
        self,
    ) -> Pointer[
        Scalar[Self.dtype], Self.origin, address_space=Self.address_space
    ]:
        """Reads the slot and returns the data pointer this handle refers to.

        Device only: both the slot and the pointer it holds are device
        addresses. Reading the slot on the host is a compile error rather than
        a runtime abort so that a host-side use of a stable-address tile is
        caught where it is written.
        """
        comptime assert is_gpu(), (
            "StableAddrEngine reads its slot on the device; loads and stores"
            " through a stable-address TileTensor belong inside the kernel"
            " that consumes it"
        )
        comptime ResultPtr = Pointer[
            Scalar[Self.dtype], Self.origin, address_space=Self.address_space
        ]
        # The slot's contents are an untracked mutable pointer; `rebind`
        # attributes it to the handle's origin and mutability.
        return rebind[ResultPtr](
            self.addr.ptr[]
            .unsafe_bitcast[Scalar[Self.dtype]]()
            .unsafe_address_space_cast[Self.address_space]()
            .unsafe_offset(self.offset)
        )


struct StableAddrEngine[*, element_width: Int = 1](TensorEngine):
    """Implements `TensorEngine` over a device graph stable address slot.

    `StableAddrEngine` is the slot-backed analogue of `DevicePointerEngine`.
    Its `StorageType` handle is a `StableStorage`: the address of a slot plus
    an element offset. The slot is graph-lifetime device memory holding one
    device data pointer; a recorded graph bakes in the slot's address and
    `DeviceGraphBuilder.place_stable()` rewrites its contents before each
    replay. Because the handle stores the slot rather than the pointer, the
    indirection is preserved across every `TileTensor` view derived from it.

    Operations that dereference storage (`load`, `store`, `copy_from`,
    `unsafe_ptr`) read the slot and are device only: they are compile errors
    on the host. Operations that only manipulate the handle (`offset`,
    `distance`, `unsafe_cast`) work on both host and device, so a tile over a
    slot can be constructed, tiled, and sliced on the host and handed to a
    kernel.

    The engine does not implement `TensorOps`; the elementwise arithmetic on
    `TileTensor` is unavailable for slot-backed tiles until it does.

    Parameters:
        element_width: Number of scalar elements per logical element. A value
            of `1` (the default) is a non-vectorized tensor; larger values
            describe a vectorized view whose logical elements are SIMD vectors.
            The handle's offset is always in scalar-element units, so
            `element_width` only sets `element_size` (and thus the tile's
            vectorized `ElementType`), exactly as for `DefaultEngine`.
    """

    comptime element_size = Self.element_width
    """Number of scalar elements per logical element (alias of `element_width`)."""

    comptime _BASE_TYPE_NAME: StaticString = "StableAddrEngine"
    """The unparameterized name of this engine."""

    comptime StorageType[
        mut: Bool,
        //,
        dtype: DType,
        origin: Origin[mut=mut],
        address_space: AddressSpace,
    ]: TrivialRegisterPassable = StableStorage[dtype, origin, address_space]
    """A `StableStorage` handle: the slot plus a scalar-element offset.

    Parameters:
        mut: The mutability of the viewed storage, inferred from `origin`.
        dtype: The element data type of the viewed storage.
        origin: The origin the view is attributed to.
        address_space: The address space of the viewed storage.
    """

    @staticmethod
    def write_type_name_to(mut writer: Some[Writer]):
        """Writes the engine type name representation to the writer.

        Args:
            writer: The `Writer` to output to.
        """
        t"StableAddrEngine[element_size={Self.element_size}]".write_to(writer)

    @doc_hidden
    @staticmethod
    @inline(.always)
    def unsafe_ptr[
        mut: Bool,
        dtype: DType,
        origin: Origin[mut=mut],
        address_space: AddressSpace,
        //,
    ](
        storage: Self.StorageType[dtype, origin, address_space],
    ) -> Pointer[
        Scalar[dtype], origin, address_space=address_space
    ]:
        """Returns the data pointer the slot holds now, advanced to this
        handle's offset.

        Device only. The result no longer goes through the slot, so it is
        valid only for the replay (or direct launch) that reads it.

        Parameters:
            mut: The mutability of the viewed storage, inferred from `origin`.
            dtype: The element data type of the viewed storage.
            origin: The origin the view is attributed to.
            address_space: The address space of the viewed storage.

        Args:
            storage: The handle to resolve.

        Returns:
            A `Pointer` to the first scalar element the handle refers to.
        """
        return storage._leaf_ptr()

    @staticmethod
    @inline(.always)
    def unsafe_cast[
        to_mut: Bool,
        //,
        to_dtype: DType,
        to_origin: Origin[mut=to_mut],
        to_address_space: AddressSpace,
    ](storage: Self.StorageType[...]) -> Self.StorageType[
        to_dtype, to_origin, to_address_space
    ]:
        """Reinterprets a handle with new type parameters.

        The slot and offset are untyped, so this rebuilds the handle under the
        new parameters without touching either. Note the offset stays in
        scalar elements of the *old* dtype; a dtype cast on a handle with a
        non-zero offset is the caller's responsibility, as for the other
        engines.

        Parameters:
            to_mut: The mutability to reinterpret the storage as.
            to_dtype: The element data type to reinterpret the storage as.
            to_origin: The origin to reinterpret the storage as.
            to_address_space: The address space to reinterpret the storage as.

        Args:
            storage: The handle to reinterpret.

        Returns:
            A handle over the same slot and offset with the new parameters.
        """
        return {storage.addr, storage.offset}

    @staticmethod
    @inline(.always)
    def load[
        dtype: DType,
        //,
        width: SIMDLength,
        alignment: Int,
        invariant: Bool = False,
        non_temporal: Bool = False,
    ](storage: Self.StorageType[mut=False, dtype, ...]) -> SIMD[dtype, width]:
        """Loads a `SIMD` value from the storage.

        Device only: reads the slot to find the data pointer.

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
        return storage._leaf_ptr().unsafe_load[
            width=width,
            alignment=alignment,
            invariant=invariant,
            non_temporal=non_temporal,
        ]()

    @staticmethod
    @inline(.always)
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
        """Loads a `SIMD` value at a scalar-element offset from the storage.

        Device only: reads the slot to find the data pointer.

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
        return storage._leaf_ptr().unsafe_load[
            width=width,
            alignment=alignment,
            invariant=invariant,
            non_temporal=non_temporal,
        ](offset)

    @staticmethod
    @inline(.always)
    def store[
        dtype: DType,
        alignment: Int,
        *,
        non_temporal: Bool = False,
    ](storage: Self.StorageType[mut=True, dtype, ...], value: SIMD[dtype, _]):
        """Stores a `SIMD` value into the storage.

        Device only: reads the slot to find the data pointer.

        Parameters:
            dtype: The element data type of the storage.
            alignment: The alignment guarantee for the store.
            non_temporal: If True, indicates the data will not be reused soon,
                allowing the hardware to bypass caches (e.g., streaming stores).

        Args:
            storage: The storage to store into.
            value: The `SIMD` value to store.
        """
        storage._leaf_ptr().unsafe_store[
            alignment=alignment, non_temporal=non_temporal
        ](value)

    @staticmethod
    @inline(.always)
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

        Device only: reads the slot to find the data pointer.

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
        storage._leaf_ptr().unsafe_store[
            alignment=alignment, non_temporal=non_temporal
        ](offset, value)

    comptime OffsetResultType[
        offset_types: TypeList[Trait=CoordLike, ...],
    ]: TensorEngine = Self
    """The storage type produced by offsetting with a given coordinate.

    Offsetting never changes the engine, so this is `Self`.

    Parameters:
        offset_types: The coordinate element types of the applied offset.
    """

    @staticmethod
    @inline(.always)
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
        """Returns a handle advanced by a number of scalar elements.

        Only the handle's offset changes; the slot is untouched, so the result
        still resolves through the slot when it is eventually dereferenced.
        Works on host and device.

        Parameters:
            offset_mut: The mutability of the storage, inferred from
                `offset_origin`.
            offset_types: The coordinate element types of `offset_coord`.
            offset_dtype: The element data type of the storage.
            offset_origin: The origin tracking the lifetime of the storage.
            offset_address_space: The address space the storage resides in.

        Args:
            storage: The handle to offset from.
            offset_coord: A flat coordinate whose components sum to the
                number of scalar elements to advance the handle by.

        Returns:
            A handle over the same slot, that many scalar elements further in.
        """
        return {
            storage.addr,
            storage.offset + Int64(_offset_elements(offset_coord)),
        }

    @staticmethod
    def distance[
        dtype: DType, address_space: AddressSpace, //
    ](
        storage: Self.StorageType[mut=False, dtype, _, address_space],
        other: Self.StorageType[mut=False, dtype, _, address_space],
    ) -> Int:
        """Returns the scalar-element distance from `other` to `storage`.

        Both handles must refer to the same slot, since the distance between
        two slots' contents is unknowable without reading them (and is not
        stable across replays anyway). Works on host and device.

        Parameters:
            dtype: The storages' `DType`.
            address_space: The storages' `AddressSpace`.

        Args:
            storage: The handle to measure the distance to.
            other: The handle to measure the distance from.

        Returns:
            The number of scalar elements separating the two handles. The
            value is positive when `storage` is ahead of `other` and negative
            when it precedes `other`.
        """
        if storage.addr.ptr != other.addr.ptr:
            abort(
                "StableAddrEngine.distance requires handles over the same"
                " stable address slot"
            )
        return Int(storage.offset - other.offset)

    @staticmethod
    @inline(.always)
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
        OtherEngine: TensorEngine,
    ](
        storage: Tuple[
            Self.StorageType[dst_dtype, self_origin, self_address_space],
            SelfLayoutType,
        ],
        other: Tuple[
            OtherEngine.StorageType[
                src_dtype, other_origin, other_address_space
            ],
            OtherLayoutType,
        ],
    ):
        """Copies the elements of `other` into `storage`, in place.

        Delegates to the shared `_copy_from` loop; destination accesses go
        through this engine's `load`/`store`, so they are device only.

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
            OtherEngine: The engine of the source. May differ from
                `Self` as long as the two engines are copy-compatible (same
                logical element size).

        Constraints:

        - Both operands must have statically known shapes with matching total
            element count.
        - Both operands must have the same logical element size.
        - Source and destination dtypes may differ; each logical element is
            cast to the destination dtype.

        Args:
            storage: A tuple of the destination storage (modified in place) and
                its layout.
            other: A tuple of the source storage and its layout.
        """
        _copy_from[
            DstEngine=Self,
            self_address_space=self_address_space,
            other_address_space=other_address_space,
        ](storage, other)


comptime StableTileTensor[dtype: DType, LayoutType: TensorLayout] = TileTensor[
    dtype,
    LayoutType,
    MutUntrackedOrigin,
    Engine=StableAddrEngine[element_width=1],
    address_space=.GENERIC,
]
"""A `TileTensor` backed by `StableAddrEngine`.

The tensor carries the slot's address, not the data pointer the slot holds,
so every view derived from it keeps resolving through the slot at the point
of use.

Parameters:
    dtype: The element data type of the tensor.
    LayoutType: The layout type of the tensor.
"""


@inline(.always)
def stable_tile_tensor[
    LayoutType: TensorLayout, //, dtype: DType
](addr: StableAddr, layout: LayoutType) -> StableTileTensor[dtype, LayoutType]:
    """Builds a `StableAddrEngine`-backed `TileTensor` over a stable address
    slot.

    The result can be constructed, tiled, and sliced on the host and passed to
    a kernel; loads and stores are device only. A graph recorded against a
    kernel that consumes the tensor bakes in the slot, so
    `DeviceGraphBuilder.place_stable()` can retarget the tensor before each
    replay without the graph being rebuilt. Nothing keeps the slot alive on the
    tile's behalf; they are assumed to live for the entire graph forward pass.

    Parameters:
        LayoutType: The layout type, inferred from `layout`.
        dtype: The element data type of the tensor.

    Args:
        addr: The stable address slot holding the tensor's data pointer.
        layout: The layout of the tensor.

    Returns:
        A slot-backed tile tensor at element offset zero.
    """
    comptime Storage = StableStorage[
        dtype, MutUntrackedOrigin, AddressSpace.GENERIC
    ]
    return StableTileTensor[dtype, LayoutType](Storage(addr, 0), layout)
