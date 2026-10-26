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
"""Defines tensors that own their storage, with the storage supplied by an
engine.

`TileTensor` borrows memory that something else owns. `OwnedTensor` is the
owner: it allocates its storage on construction and hands out `TileTensor`
views whose origin is tied to it. An `OwnedTensorEngine` decides what the
storage is and which `TensorEngine` its views use, so one `OwnedTensor` type
covers several kinds of memory. `GPUEngine` is the shared-memory engine, and
`GPUTensor` names an `OwnedTensor` that uses it.
"""

from layout import TensorLayout, row_major
from layout.tensor_engine import DefaultEngine, TensorEngine
from std.memory import stack_allocation
from std.sys import align_of


trait OwnedTensorEngine:
    """Supplies the storage behind an `OwnedTensor` and views of that storage.

    A conforming engine names the owning storage an `OwnedTensor` holds and
    the `TensorEngine` of the `TileTensor` views built from that tensor.
    `get_storage_view` turns the owned storage into that engine's storage
    handle.
    """

    comptime StorageType[
        dtype: DType, address_space: AddressSpace, LayoutType: TensorLayout
    ]: Deinitable & Defaultable
    """The owning storage for a tensor with the given element type and layout.

    `OwnedTensor` default-constructs this type to allocate its elements, so
    the layout must be enough to size the storage.

    Parameters:
        dtype: The element data type of the storage.
        address_space: The address space the storage resides in.
        LayoutType: The layout of the tensor the storage backs.
    """

    comptime ViewEngine: TensorEngine
    """The engine of a `TileTensor` built from an `OwnedTensor`."""

    @staticmethod
    def get_storage_view[
        result_mut: Bool,
        dtype: DType,
        address_space: AddressSpace,
        result_origin: Origin[mut=result_mut],
        //,
    ](
        ref storage: Self.StorageType[dtype, address_space, _],
    ) -> Self.ViewEngine.StorageType[dtype, result_origin, address_space]:
        """Returns a `ViewEngine` storage handle to owned storage.

        The caller chooses `result_origin`, either through the return type it
        expects or by passing it by keyword. It should be the origin of the
        `OwnedTensor` holding `storage`, so the view can't outlive the owner.

        Parameters:
            result_mut: The mutability of the view, inferred from
                `result_origin`.
            dtype: The element data type of the storage.
            address_space: The address space the storage resides in.
            result_origin: The origin the view borrows from.

        Args:
            storage: The owned storage to view.

        Returns:
            A handle to `storage` with origin `result_origin`.
        """
        ...


struct OwnedTensor[
    dtype: DType,
    address_space: AddressSpace,
    LayoutType: TensorLayout,
    *,
    Engine: OwnedTensorEngine,
](
    Movable where conforms_to(
        Engine.StorageType[dtype, address_space, LayoutType], Movable
    )
):
    """Owns the storage for a tensor whose layout is known at compile time.

    Parameters:
        dtype: The element data type of the tensor.
        address_space: The address space the storage resides in.
        LayoutType: The layout of the tensor. Constrained to layouts whose
            dimensions are all known at compile time.
        Engine: The engine that supplies the storage.

    Construction allocates storage through `Engine`, and access goes through a
    `TileTensor` built from the tensor, whose origin is tied to it. The tensor
    is `Movable` exactly when its storage is.

    Examples:

    ```mojo
    from layout import TileTensor, row_major
    from layout.owned_tensor import GPUTensor

    def kernel():
        var owned = GPUTensor[.float32, .SHARED](row_major[4, 8]())
        var tile = TileTensor(owned)
        tile[1, 2] = 7.0
    ```
    """

    comptime Storage = Self.Engine.StorageType[
        Self.dtype, Self.address_space, Self.LayoutType
    ]
    """The owning storage `Engine` supplies for this tensor."""

    var _storage: Self.Storage
    var layout: Self.LayoutType
    """The layout that maps coordinates to offsets in the storage."""

    @inline(.always)
    def __init__(out self, layout: Self.LayoutType):
        """Allocates storage for a tensor with the given layout.

        The elements are not initialized.

        Args:
            layout: The layout of the tensor.

        Constraints:
            Every dimension of `LayoutType` must be known at compile time.
        """
        comptime assert Self.LayoutType.all_dims_known
        self._storage = {}
        self.layout = layout


struct OwnedGPUStorage[
    dtype: DType,
    address_space: AddressSpace,
    LayoutType: TensorLayout,
    alignment: Int = align_of[dtype](),
](Defaultable, not Movable):
    """Holds the elements of a compact layout in a GPU stack allocation.

    Parameters:
        dtype: The element data type.
        address_space: The address space to allocate in.
        LayoutType: The layout of the tensor the storage backs. Constrained to
            compact layouts, whose offsets cover exactly one slot per element.
        alignment: The alignment of the underlying memory.

    The memory comes from `stack_allocation`. Only a shared-memory allocation
    outlives the constructor, so use this with `AddressSpace.SHARED` inside a
    kernel. Every instance in a block refers to the same shared allocation.
    """

    var _ptr: Pointer[
        Scalar[Self.dtype], MutUntrackedOrigin, address_space=Self.address_space
    ]

    def __init__(out self):
        """Allocates one uninitialized slot per element of `LayoutType`.

        Constraints:
            `LayoutType` must be compact: its cosize must equal its number of
            elements.
        """
        comptime assert (
            Self.LayoutType.static_cosize == Self.LayoutType.static_product
        ), "OwnedGPUStorage requires a compact layout"
        self._ptr = stack_allocation[
            Self.LayoutType.static_product,
            Scalar[Self.dtype],
            address_space=Self.address_space,
            alignment=Self.alignment,
        ]()

    def unsafe_ptr(
        ref self,
    ) -> Pointer[
        Scalar[Self.dtype], origin_of(self), address_space=Self.address_space
    ]:
        """Returns a pointer to the first element.

        Returns:
            A pointer whose origin and mutability are those of `self`.
        """
        return self._ptr.unsafe_mut_cast[
            origin_of(self).mut
        ]().unsafe_origin_cast[origin_of(self)]()


struct OwnedGPUEngine(OwnedTensorEngine):
    """Backs an `OwnedTensor` with GPU shared memory.

    Storage is a `OwnedGPUStorage`, so the layout must be compact, and views use
    `DefaultEngine`, whose storage handle is a plain `Pointer`.
    """

    comptime StorageType[
        dtype: DType, address_space: AddressSpace, LayoutType: TensorLayout
    ]: Deinitable & Defaultable = OwnedGPUStorage[
        dtype, address_space, LayoutType
    ]
    """A `OwnedGPUStorage` for a tensor with layout `LayoutType`.

    Parameters:
        dtype: The element data type of the storage.
        address_space: The address space the storage resides in.
        LayoutType: The layout of the tensor the storage backs.
    """

    comptime ViewEngine: TensorEngine = DefaultEngine[element_width=1]
    """Views shared memory through a plain pointer."""

    @staticmethod
    def get_storage_view[
        result_mut: Bool,
        dtype: DType,
        address_space: AddressSpace,
        result_origin: Origin[mut=result_mut],
        //,
    ](
        ref storage: Self.StorageType[dtype, address_space, _],
    ) -> Self.ViewEngine.StorageType[dtype, result_origin, address_space]:
        """Returns a pointer to the storage with origin `result_origin`.

        Parameters:
            result_mut: The mutability of the view, inferred from
                `result_origin`.
            dtype: The element data type of the storage.
            address_space: The address space the storage resides in.
            result_origin: The origin the view borrows from.

        Args:
            storage: The owned storage to view.

        Returns:
            A pointer to the first element of `storage`.
        """
        return (
            storage.unsafe_ptr()
            .unsafe_mut_cast[result_mut]()
            .unsafe_origin_cast[result_origin]()
        )


comptime OwnedGPUTensor[
    dtype: DType, address_space: AddressSpace, LayoutType: TensorLayout
] = OwnedTensor[dtype, address_space, LayoutType, Engine=OwnedGPUEngine]
"""Names an `OwnedTensor` whose storage comes from `GPUEngine`.

Parameters:
    dtype: The element data type of the tensor.
    address_space: The address space the storage resides in, normally
        `AddressSpace.SHARED`.
    LayoutType: The layout of the tensor.
"""
