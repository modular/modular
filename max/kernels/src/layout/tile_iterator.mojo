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
"""A `TileTensor` iterator that walks a buffer one tile at a time.

`TileTensorIter` steps through a buffer of equally shaped tiles. It
holds a pointer, a fixed tile layout and a step between consecutive tiles, and
dereferences to a `TileTensor` view of the current tile. It supports the
circular multi-stage shared-memory buffers used by software-pipelined kernels
and the masked global-memory tiles whose valid row count is only known at run
time.
"""

from .tile_layout import TensorLayout
from .tile_tensor import TileTensor, _get_index_type


struct TileTensorIter[
    mut: Bool,
    //,
    dtype: DType,
    LayoutType: TensorLayout,
    origin: Origin[mut=mut],
    /,
    *,
    address_space: AddressSpace = .GENERIC,
    circular: Bool = False,
    masked: Bool = False,
    linear_idx_type: DType = _get_index_type[LayoutType](address_space),
](TrivialRegisterPassable):
    """Iterates over equally shaped tiles of a buffer.

    Every tile shares `LayoutType`; the iterator advances by `stride` elements
    from one tile to the next. A circular iterator wraps back to the start of
    the buffer once the offset reaches `bound`.

    Parameters:
        mut: Whether the iterator allows mutation of the underlying data.
        dtype: The data type of the tensor elements.
        LayoutType: The layout of each tile the iterator yields.
        origin: The origin of the underlying pointer.
        address_space: The memory address space of the buffer.
        circular: Whether iteration wraps around at `bound`.
        masked: Whether the tiles carry a runtime valid-row count in
            `num_valid_rows`. The iterator itself never clips the tile
            layout; consumers that copy a masked tile pass the count as the
            copy's explicit bound.
        linear_idx_type: Signed integer type of offsets into the buffer.
    """

    comptime TileType = TileTensor[
        Self.dtype,
        Self.LayoutType,
        Self.origin,
        address_space=Self.address_space,
        linear_idx_type=Self.linear_idx_type,
    ]
    """The `TileTensor` type produced by dereferencing the iterator."""

    comptime linear_uint_type = Scalar[Self.linear_idx_type]
    """The integer type of `offset`, `stride` and `bound`."""

    var ptr: Pointer[
        Scalar[Self.dtype], Self.origin, address_space=Self.address_space
    ]
    """Pointer to the start of the buffer being iterated."""

    var tile_layout: Self.LayoutType
    """Layout shared by every tile."""

    var offset: Scalar[Self.linear_idx_type]
    """Element offset of the current tile from `ptr`."""

    var stride: Scalar[Self.linear_idx_type]
    """Element distance between consecutive tiles."""

    var bound: Scalar[Self.linear_idx_type]
    """Buffer size in elements; the wrap-around point of a circular iterator."""

    var num_valid_rows: Int
    """Valid rows of each tile when `masked`; unused otherwise."""

    @always_inline
    def __init__(
        out self,
        ptr: Pointer[
            Scalar[Self.dtype], Self.origin, address_space=Self.address_space
        ],
        tile_layout: Self.LayoutType,
        *,
        bound: Int,
        stride: Int,
        offset: Int = 0,
        num_valid_rows: Int = 0,
    ):
        """Creates an iterator over a buffer.

        Args:
            ptr: Pointer to the start of the buffer.
            tile_layout: Layout of every tile.
            bound: Buffer size in elements; the wrap-around point when
                `circular`.
            stride: Element distance between consecutive tiles.
            offset: Element offset of the first tile.
            num_valid_rows: Valid rows of each tile; only meaningful when
                `masked`.
        """
        comptime assert (
            Self.linear_idx_type.is_signed()
        ), "TileTensorIter requires a signed linear index type."
        self.ptr = ptr
        self.tile_layout = tile_layout
        self.offset = Self.linear_uint_type(offset)
        self.stride = Self.linear_uint_type(stride)
        self.bound = Self.linear_uint_type(bound)
        self.num_valid_rows = num_valid_rows

    @always_inline
    def __init__(
        out self,
        ptr: Pointer[
            Scalar[Self.dtype], Self.origin, address_space=Self.address_space
        ],
        tile_layout: Self.LayoutType,
        *,
        bound: Int,
        num_valid_rows: Int = 0,
    ) where Self.LayoutType.all_dims_known:
        """Creates an iterator whose tiles are laid out back to back.

        Args:
            ptr: Pointer to the start of the buffer.
            tile_layout: Layout of every tile; its element count is the
                stride.
            bound: Buffer size in elements; the wrap-around point when
                `circular`.
            num_valid_rows: Valid rows of each tile; only meaningful when
                `masked`.
        """
        self = Self(
            ptr,
            tile_layout,
            bound=bound,
            stride=Self.LayoutType.static_product,
            num_valid_rows=num_valid_rows,
        )

    @always_inline
    def __init__(
        out self,
        tensor: TileTensor[
            Self.dtype,
            _,
            Self.origin,
            address_space=Self.address_space,
            ...,
        ],
        tile_layout: Self.LayoutType,
        *,
        bound: Int,
        stride: Int,
        offset: Int = 0,
        num_valid_rows: Int = 0,
    ):
        """Creates an iterator starting at the base of a tensor's storage.

        Args:
            tensor: Tensor whose base pointer starts the buffer; pass a tile or
                view to start elsewhere.
            tile_layout: Layout of every tile.
            bound: Buffer size in elements; the wrap-around point when
                `circular`.
            stride: Element distance between consecutive tiles.
            offset: Element offset of the first tile from the start of the
                buffer.
            num_valid_rows: Valid rows of each tile; only meaningful when
                `masked`.
        """
        self = Self(
            tensor.unsafe_ptr(),
            tile_layout,
            bound=bound,
            stride=stride,
            offset=offset,
            num_valid_rows=num_valid_rows,
        )

    @always_inline
    def __init__(
        out self,
        tensor: TileTensor[
            Self.dtype,
            _,
            Self.origin,
            address_space=Self.address_space,
            ...,
        ],
        tile_layout: Self.LayoutType,
        *,
        bound: Int,
        num_valid_rows: Int = 0,
    ) where Self.LayoutType.all_dims_known:
        """Creates an iterator starting at the base of a tensor's storage, with
        tiles laid out back to back.

        Args:
            tensor: Tensor whose base pointer starts the buffer; pass a tile or
                view to start elsewhere.
            tile_layout: Layout of every tile; its element count is the
                stride.
            bound: Buffer size in elements; the wrap-around point when
                `circular`.
            num_valid_rows: Valid rows of each tile; only meaningful when
                `masked`.
        """
        self = Self(
            tensor,
            tile_layout,
            bound=bound,
            stride=Self.LayoutType.static_product,
            num_valid_rows=num_valid_rows,
        )

    @always_inline
    def get(self) -> Self.TileType:
        """Returns the tile at the current position.

        Returns:
            A `TileTensor` view of the current tile.
        """
        return Self.TileType(
            ptr=self.ptr.unsafe_offset(Int(self.offset)),
            layout=self.tile_layout,
        )

    @always_inline
    def __getitem__(self) -> Self.TileType:
        """Returns the tile at the current position.

        Returns:
            A `TileTensor` view of the current tile.
        """
        return self.get()

    @always_inline
    def _incr(mut self):
        """Advances by one tile without bounds checking.

        A circular iterator wraps with a conditional subtract, so it must
        never be stepped by more than `bound`.
        """
        self.offset += self.stride

        comptime if Self.circular:
            self.offset = (
                self.offset - self.bound if self.offset
                >= self.bound else self.offset
            )

    @always_inline
    def __iadd__(mut self, rhs: Int):
        """Advances by `rhs` tiles.

        A circular iterator wraps with a modulo, so `rhs` may exceed the
        number of tiles in the buffer.

        Args:
            rhs: The number of tiles to advance.
        """
        self.offset += Self.linear_uint_type(rhs) * self.stride

        comptime if Self.circular:
            self.offset = self.offset % self.bound

    @always_inline
    def next(self, rhs: Int = 1) -> Self:
        """Returns an iterator `rhs` tiles ahead.

        Args:
            rhs: The number of tiles to advance.

        Returns:
            A copy of this iterator advanced by `rhs` tiles.
        """
        var result = self
        result += rhs
        return result

    @always_inline
    def next_unsafe(self, rhs: Int = 1) -> Self:
        """Returns an iterator `rhs` tiles ahead, wrapping at most once.

        Cheaper than `next` for a circular iterator: the wrap is a single
        conditional subtract, so `rhs * stride` must be less than `bound`.

        Args:
            rhs: The number of tiles to advance.

        Returns:
            A copy of this iterator advanced by `rhs` tiles.
        """
        var result = self
        result.offset += Self.linear_uint_type(rhs) * self.stride

        comptime if Self.circular:
            result.offset = (
                result.offset - self.bound if result.offset
                >= self.bound else result.offset
            )
        return result

    comptime BitcastType[target_dtype: DType] = TileTensorIter[
        target_dtype,
        Self.LayoutType,
        Self.origin,
        address_space=Self.address_space,
        circular=Self.circular,
        masked=Self.masked,
        linear_idx_type=Self.linear_idx_type,
    ]
    """The iterator type produced by `bitcast`.

    Parameters:
        target_dtype: The element type the buffer is viewed as.
    """

    @always_inline
    def bitcast[target_dtype: DType](self) -> Self.BitcastType[target_dtype]:
        """Reinterprets the buffer as `target_dtype`, keeping the tile layout,
        offset, stride and bound.

        Parameters:
            target_dtype: The element type to view the buffer as.

        Returns:
            An iterator over the same memory and position.
        """
        return Self.BitcastType[target_dtype](
            self.ptr.unsafe_bitcast[Scalar[target_dtype]](),
            self.tile_layout,
            bound=Int(self.bound),
            stride=Int(self.stride),
            offset=Int(self.offset),
            num_valid_rows=self.num_valid_rows,
        )

    comptime ReshapeType[NewLayout: TensorLayout] = TileTensorIter[
        Self.dtype,
        NewLayout,
        Self.origin,
        address_space=Self.address_space,
        circular=Self.circular,
        masked=Self.masked,
        linear_idx_type=Self.linear_idx_type,
    ]
    """The iterator type produced by `reshape`.

    Parameters:
        NewLayout: The layout of each tile of the result.
    """

    @always_inline
    def reshape[
        NewLayout: TensorLayout
    ](self, new_layout: NewLayout) -> Self.ReshapeType[NewLayout]:
        """Views each tile under `new_layout`, keeping the offset, stride and
        bound.

        Parameters:
            NewLayout: The layout of each tile of the result.

        Args:
            new_layout: The layout of each tile of the result.

        Returns:
            An iterator over the same memory and position.
        """
        return Self.ReshapeType[NewLayout](
            self.ptr,
            new_layout,
            bound=Int(self.bound),
            stride=Int(self.stride),
            offset=Int(self.offset),
            num_valid_rows=self.num_valid_rows,
        )
