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
"""Transitional tile-codegen scaffolding: the terminal store and the epilogue
operation of a tile kernel's output.

`_TileStoreConsumer` copies each fragment into the output. The legacy tile
consumer and operation forward each tile to the fused epilogue of an output
declared with a legacy tile trait. They live here rather than on the traits
because `TileConsumer` and `TileOperation` are `linalg` traits.

Expected to be removed with the rest of the tile path's scaffolding once it is
productionized (see GEX-3919).
"""

from layout import Coord, TensorLayout, TileTensor, coord_to_index_list
from layout.layout import Layout as _NewLayout
from layout.tile_io import GenericToLocalTileCopier, LocalToGenericTileCopier
from std.utils import IndexList
from linalg.utils import TileConsumer, TileOperation
from extensibility import _LegacyTileComputable, _LegacyTileStorable


@fieldwise_init
struct _TileStoreConsumer[
    out_dtype: DType,
    OutLayout: TensorLayout,
](TileConsumer):
    """Terminal `TileConsumer` that copies each fragment into the output."""

    comptime src_address_space = AddressSpace.LOCAL

    var output: TileTensor[Self.out_dtype, Self.OutLayout, MutUntrackedOrigin]

    @inline(.always)
    def __call__[
        dtype: DType,
        LayoutType: TensorLayout,
    ](
        ref self,
        tile_coord: Coord,
        tile: TileTensor[
            dtype, LayoutType, ..., address_space=Self.src_address_space
        ],
        thread_layout: _NewLayout,
    ) -> None:
        comptime assert type_of(
            thread_layout
        ).all_dims_known, "TileConsumer thread_layout must be statically known"
        comptime assert tile_coord.is_flat, "Tile coordinate must be flat."
        # `tile` is this thread's fragment; the output tile spans the whole
        # thread layout.
        comptime TM = type_of(thread_layout).static_shape[0]
        comptime TN = type_of(thread_layout).static_shape[1]
        var elem = rebind[IndexList[2]](coord_to_index_list(tile_coord))
        var out_tile = self.output.tile[TM, TN](
            Coord(elem[0] // TM, elem[1] // TN)
        )
        LocalToGenericTileCopier[type_of(thread_layout)()]().copy(
            out_tile, tile
        )


# TODO(GEX-4340): the consumer and operation over an output declared with a
# legacy tile trait. Remove with the legacy tile traits.


@fieldwise_init
struct _LegacyTileStoreConsumer[
    OutType: _LegacyTileStorable & TrivialRegisterPassable,
](TileConsumer):
    """Terminal `TileConsumer` over an output's legacy tile store fusion.

    Holds the output's device value and forwards each tile to its `store`,
    which runs the epilogue and owns the write. Build it on the device: the
    host value cannot call `store`.
    """

    comptime src_address_space = AddressSpace.GENERIC

    var output: Self.OutType

    @inline(.always)
    def __call__[
        dtype: DType,
        LayoutType: TensorLayout,
    ](
        ref self,
        tile_coord: Coord,
        tile: TileTensor[
            dtype, LayoutType, ..., address_space=Self.src_address_space
        ],
        thread_layout: _NewLayout,
    ) -> None:
        comptime assert type_of(
            thread_layout
        ).all_dims_known, "TileConsumer thread_layout must be statically known"
        comptime assert tile_coord.is_flat, "Tile coordinate must be flat."

        var copier = GenericToLocalTileCopier[type_of(thread_layout)()]()
        var store_copier = LocalToGenericTileCopier[type_of(thread_layout)()]()
        comptime TM = type_of(thread_layout).static_shape[0]
        comptime TN = type_of(thread_layout).static_shape[1]
        var elem = rebind[IndexList[2]](coord_to_index_list(tile_coord))
        var ridx = IndexList[2](elem[0] // TM, elem[1] // TN)
        self.output.store[
            dtype, 2, LayoutType, type_of(copier), type_of(store_copier)
        ](
            ridx,
            copier,
            store_copier,
            rebind[
                TileTensor[
                    dtype,
                    LayoutType,
                    MutAnyOrigin,
                    address_space=Self.src_address_space,
                ]
            ](tile),
        )


@fieldwise_init
struct _LegacyTileComputeOp[
    OutType: _LegacyTileComputable & TrivialRegisterPassable,
](TileOperation):
    """`TileOperation` over an output's legacy tile compute fusion.

    Holds the output's device value and forwards each tile to its `compute`,
    which returns the transformed tile for the kernel to store. Build it on
    the device: the host value cannot call `compute`.
    """

    comptime src_address_space = AddressSpace.GENERIC

    var output: Self.OutType

    @inline(.always)
    def __call__[
        dtype: DType,
        LayoutType: TensorLayout,
    ](
        mut self,
        tile_coord: Coord,
        tile: TileTensor[
            dtype,
            LayoutType,
            MutAnyOrigin,
            ...,
            address_space=Self.src_address_space,
        ],
        thread_layout: _NewLayout,
    ) -> type_of(tile):
        comptime assert type_of(
            thread_layout
        ).all_dims_known, "TileOperation thread_layout must be statically known"
        comptime assert tile_coord.is_flat, "Tile coordinate must be flat."
        var copier = GenericToLocalTileCopier[type_of(thread_layout)()]()
        comptime TM = type_of(thread_layout).static_shape[0]
        comptime TN = type_of(thread_layout).static_shape[1]
        var elem = rebind[IndexList[2]](coord_to_index_list(tile_coord))
        var ridx = IndexList[2](elem[0] // TM, elem[1] // TN)
        return rebind[type_of(tile)](
            self.output.compute[dtype, 2, LayoutType, type_of(copier)](
                ridx,
                copier,
                rebind[
                    TileTensor[
                        dtype,
                        LayoutType,
                        MutAnyOrigin,
                        address_space=Self.src_address_space,
                    ]
                ](tile),
            )
        )
