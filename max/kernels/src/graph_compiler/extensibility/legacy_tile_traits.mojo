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
"""Interim tensor-argument traits for tile-based epilogue fusion.

TODO(GEX-4340): These traits are a stopgap. They replace the tile IOSpecs of
the tile-fused kernels until the tile epilogue interface of GEX-4467 replaces
them, and they are deleted then. The graph compiler accepts them only on a
fixed list of kernel registrations, so do not use them in a new kernel.

Each lane has two traits, as `StoreAccess` and `Storable` do. The `execute`
signature names the first one, which has no methods. The kernel body asserts
the second one, which adds the method the kernel calls:

- Tile store: `_LegacyTileStoreAccess` / `_LegacyTileStorable`. `store` runs
  the whole epilogue on one tile, including the write. The argument does not
  conform to `TileTensorable`.
- Tile compute: `_LegacyTileComputeOutput` / `_LegacyTileComputable`.
  `compute` transforms one tile and returns it; the kernel stores the result
  through `to_tile_tensor`. The signature must also bind `TileTensorable`.

Each capability also gives the parts of the argument, for a kernel whose
launch cannot take the argument as one value: `tile_fusion()` returns the
generated fusion struct, and `storage()` returns the output of the store lane.
That kernel launches the parts separately and builds its consumer again on
the device. A kernel that calls `store` or `compute` does not use them.
"""

from std.utils import IndexList

from layout import MutTileTensor, TensorLayout, TileTensor
from layout.tile_io import TileCopier

from .managed_tensor_slice import ComputeOutputFusionTile, OutputFusionTile
from .tensor_arg_traits import Tensor


trait _LegacyTileStoreAccess(Tensor):
    """Marker trait: this output takes its fused epilogue one tile at a time,
    and the epilogue owns the write.

    Declare it as `Some[Output & Fused & _LegacyTileStoreAccess]` and refine to
    `_LegacyTileStorable` in the body. An output with nothing fused arrives the
    same way, with an epilogue that only writes the tile.

    TODO(GEX-4340): Interim; see the module docstring.
    """

    pass


trait _LegacyTileStorable(_LegacyTileStoreAccess):
    """`_LegacyTileStoreAccess` refined with the callable `store` method.

    A kernel signature must not declare it; the graph compiler rejects one that
    does.
    """

    comptime LayoutType: TensorLayout
    comptime TileFusion: OutputFusionTile

    def store[
        dtype: DType,
        rank: Int,
        LayoutType: TensorLayout,
        Copier: TileCopier,
        StoreCopier: TileCopier,
    ](
        self,
        tile_coords: IndexList[rank],
        copier: Copier,
        store_copier: StoreCopier,
        val: MutTileTensor[dtype, LayoutType, _],
    ):
        """Runs the fused epilogue on one tile and writes the result.

        Parameters:
            dtype: Element type of the tile.
            rank: Rank of the tile coordinates.
            LayoutType: Layout of the tile.
            Copier: Copier that loads the epilogue's other inputs.
            StoreCopier: Copier that writes the result to the output.

        Args:
            tile_coords: Index of the tile in the output, in tiles.
            copier: Loads the epilogue's other inputs for this tile.
            store_copier: Writes the transformed tile to the output.
            val: The tile the kernel computed.
        """
        ...

    def tile_fusion(self) -> Self.TileFusion:
        """Returns the generated fusion struct, whose `store` runs the epilogue.

        Returns:
            The fusion struct, over pointer-backed captures.
        """
        ...

    def storage(
        self,
    ) -> TileTensor[Self.dtype, Self.LayoutType, MutUntrackedOrigin]:
        """Returns the output tensor, for a launch that passes it separately.

        Only give it to the launch and to the fusion's `store`. The epilogue
        owns the write.

        Returns:
            A pointer-backed tile over the whole output.
        """
        ...


trait _LegacyTileComputeOutput(Tensor):
    """Marker trait: this output takes its fused epilogue as a tile transform,
    and the kernel stores the result.

    Declare it as
    `Some[Output & Fused & _LegacyTileComputeOutput & TileTensorable]` and
    refine to `_LegacyTileComputable` in the body. An output with nothing fused
    arrives the same way, with an identity `compute`.

    Unlike `_LegacyComputeOutput`, there is no store fallback: the graph
    compiler does not fuse an epilogue that changes the dtype, changes the
    store index, or stores more than once.

    TODO(GEX-4340): Interim; see the module docstring.
    """

    pass


trait _LegacyTileComputable(_LegacyTileComputeOutput):
    """`_LegacyTileComputeOutput` refined with the callable `compute` method.

    A kernel signature must not declare it; the graph compiler rejects one that
    does.
    """

    comptime TileFusion: ComputeOutputFusionTile

    def compute[
        dtype: DType,
        rank: Int,
        LayoutType: TensorLayout,
        Copier: TileCopier,
    ](
        self,
        tile_coords: IndexList[rank],
        copier: Copier,
        val: MutTileTensor[dtype, LayoutType, _],
    ) -> type_of(val):
        """Applies the fused epilogue to one tile.

        Parameters:
            dtype: Element type of the tile.
            rank: Rank of the tile coordinates.
            LayoutType: Layout of the tile.
            Copier: Copier that loads the epilogue's other inputs.

        Args:
            tile_coords: Index of the tile in the output, in tiles.
            copier: Loads the epilogue's other inputs for this tile.
            val: The tile the kernel computed.

        Returns:
            The transformed tile; the kernel stores it.
        """
        ...

    def tile_fusion(self) -> Self.TileFusion:
        """Returns the generated fusion struct, whose `compute` transforms one
        tile.

        Returns:
            The fusion struct, over pointer-backed captures.
        """
        ...
