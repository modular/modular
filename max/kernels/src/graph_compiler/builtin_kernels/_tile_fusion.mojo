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
"""Transitional tile-codegen scaffolding: the terminal store of a tile kernel's
output, the tile analog of `ManagedTensorSlice._fused_store`.

`_tile_store_consumer` returns the `TileConsumer` that owns a kernel's final
store: the bound store fusion when there is one, otherwise a direct copy into
the output. A tile kernel always binds that consumer, so it never builds a
`TileTensor` over a store-fused output. The helper lives here rather than on
the slice because `TileConsumer` is a `linalg` trait.

Expected to be removed with the rest of the tile path's scaffolding once it is
productionized (see GEX-3919).
"""

from layout import Coord, TensorLayout, TileTensor, coord_to_index_list
from layout.tile_layout import Layout as _NewLayout
from layout.tile_io import GenericToLocalTileCopier, LocalToGenericTileCopier
from std.utils import IndexList
from linalg.utils import TileConsumer
from extensibility import ManagedTensorSlice, OutputFusionTile


@fieldwise_init
struct _OutputFusionTileConsumer[
    FusionType: OutputFusionTile,
](TileConsumer):
    """Terminal `TileConsumer` over an output's tile store fusion.

    Forwards the per-tile epilogue to the fusion's `store`, which owns the
    write.
    """

    comptime src_address_space = AddressSpace.GENERIC

    var fusion: Self.FusionType

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
        self.fusion.store[
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


@inline(.always)
def _tile_store_consumer(
    output: ManagedTensorSlice[mut=True, ...],
    out result: (
        _OutputFusionTileConsumer[type_of(output).OutFusionTile] if type_of(
            output
        )._has_output_fusion_tile else _TileStoreConsumer[
            type_of(output).dtype, type_of(output).RuntimeLayout
        ]
    ),
):
    """Returns the consumer that owns the final store into `output`.

    Args:
        output: The kernel's output.

    Returns:
        The store fusion's consumer when `output` has one, otherwise a
        consumer that copies into `output` directly.
    """
    comptime if type_of(output)._has_output_fusion_tile:
        result = rebind[type_of(result)](
            _OutputFusionTileConsumer(output.output_fusion_tile)
        )
    else:
        result = rebind[type_of(result)](
            _TileStoreConsumer(output.to_tile_tensor())
        )
