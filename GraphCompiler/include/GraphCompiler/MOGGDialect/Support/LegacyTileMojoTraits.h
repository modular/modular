//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the MAX License:
// https://modular.com/legal/max-license
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//
//
// TODO(GEX-4340): Mojo source emitted once per module for the interim legacy
// tile traits: the fused output arguments that wrap a tile fusion struct.
// Delete this file with the traits when GEX-4467 lands. The removal checklist
// is GraphCompiler/docs/LegacyTileTraitsRemoval.md.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MOGGDIALECT_SUPPORT_LEGACYTILEMOJOTRAITS_H
#define GENERICML_GRAPHCOMPILER_MOGGDIALECT_SUPPORT_LEGACYTILEMOJOTRAITS_H

#include "llvm/ADT/StringRef.h"

namespace M::MOGG {

inline constexpr llvm::StringLiteral kStructMoggLegacyTileStoreFusionTensor =
    "_MoggLegacyTileStoreFusionTensor";
inline constexpr llvm::StringLiteral
    kStructMoggLegacyTileTransformFusionTensor =
        "_MoggLegacyTileTransformFusionTensor";

// The tile versions of `_MoggOutputStoreFusionTensor` and
// `_MoggTransformFusionTensor` (MojoTraits.h). Each one holds the emitted tile
// fusion struct and forwards `store` or `compute` to it. As in MojoTraits.h,
// `_access_enabled` stops these calls on the GPU host value.
//
// The transform wrapper also holds the output buffer. `to_tile_tensor` works
// on the host too, because the kernel gets the buffer there to pass it to the
// launch.
static constexpr llvm::StringLiteral kLegacyTileFusionDefs = R"mojo(
from layout import Coord, MutTileTensor, TensorLayout, TileTensor
from layout.tile_io import TileCopier
from std.builtin.device_passable import DevicePassable, DeviceTypeEncoder
from std.os import abort
from std.utils import IndexList
from extensibility import (
    ComputeOutputFusionTile,
    Fused,
    Output,
    OutputFusionTile,
    Tensor,
    TileTensorable,
    UnsafeFusedView,
    _LegacyTileComputable,
    _LegacyTileComputeOutput,
    _LegacyTileStorable,
    _LegacyTileStoreAccess,
)


struct _MoggLegacyTileStoreFusionTensor[
    _dtype: DType,
    _LayoutType: TensorLayout,
    _alignment: Int,
    _TileFusion: OutputFusionTile,
    _device_passable: Bool,
    _access_enabled: Bool,
](
    Copyable,
    DevicePassable where _device_passable,
    Fused,
    Movable,
    Output,
    Tensor,
    TrivialRegisterPassable,
    UnsafeFusedView,
    _LegacyTileStorable,
    _LegacyTileStoreAccess,
):
    comptime dtype = Self._dtype
    comptime LayoutType = Self._LayoutType
    comptime rank = Self.LayoutType.rank
    comptime alignment = Self._alignment
    comptime device_passable = Self._device_passable
    comptime access_enabled = Self._access_enabled
    comptime TileFusion = Self._TileFusion
    comptime _shape_types = Self.LayoutType._shape_types
    comptime __shape_types = Self.LayoutType.__shape_types
    var _tile_fusion: Self._TileFusion
    var _layout: Self.LayoutType

    comptime device_type: AnyType = _MoggLegacyTileStoreFusionTensor[
        Self._dtype,
        Self._LayoutType,
        Self._alignment,
        Self._TileFusion,
        False,
        True,
    ]

    comptime FusedViewType: TrivialRegisterPassable = Self.device_type

    @inline(.always)
    def __init__(out self, tile_fusion: Self._TileFusion, layout: Self.LayoutType):
        self._tile_fusion = tile_fusion
        self._layout = layout

    @inline(.always)
    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ) where Self._device_passable:
        comptime fusion_offset = reflect[Self.device_type].field_offset[
            index=0, target=encoder.target()
        ]()
        encoder.encode_fields[Self._TileFusion](
            self._tile_fusion,
            target.unsafe_bitcast[UInt8]()
            .unsafe_offset(fusion_offset)
            .unsafe_bitcast[NoneType](),
        )
        comptime layout_offset = reflect[Self.device_type].field_offset[
            index=1, target=encoder.target()
        ]()
        encoder.encode(
            self._layout,
            target.unsafe_bitcast[UInt8]()
            .unsafe_offset(layout_offset)
            .unsafe_bitcast[NoneType](),
        )

    @staticmethod
    def get_type_name() -> String where Self._device_passable:
        return String("_MoggLegacyTileStoreFusionTensor")

    @inline(.always)
    def shape(self) -> Coord[*Self._shape_types]:
        return self._layout.shape_coord()

    @inline(.always)
    def tile_fusion(self) -> Self.TileFusion:
        return self._tile_fusion

    # A GPU kernel calls `store`, and the fusion struct writes the output, so
    # this wrapper holds no output buffer.
    @inline(.always)
    def storage(
        self,
    ) -> TileTensor[Self.dtype, Self.LayoutType, MutUntrackedOrigin]:
        abort("a GPU legacy tile store output has no separate storage")

    @inline(.always)
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
        comptime assert Self._access_enabled, (
            "a fused tile store is only valid on the projected device value;"
            " the device-passable host proxy cannot store directly"
        )
        self._tile_fusion.store[dtype, rank, LayoutType, Copier, StoreCopier](
            tile_coords, copier, store_copier, val
        )


struct _MoggLegacyTileTransformFusionTensor[
    _dtype: DType,
    _LayoutType: TensorLayout,
    _alignment: Int,
    _mut: Bool,
    _TileFusion: ComputeOutputFusionTile,
    _device_passable: Bool,
    _access_enabled: Bool,
](
    Copyable,
    DevicePassable where _device_passable,
    Fused,
    Movable,
    Output,
    TileTensorable,
    TrivialRegisterPassable,
    UnsafeFusedView,
    _LegacyTileComputable,
    _LegacyTileComputeOutput,
):
    comptime dtype = Self._dtype
    comptime LayoutType = Self._LayoutType
    comptime rank = Self.LayoutType.rank
    comptime alignment = Self._alignment
    comptime mut = Self._mut
    comptime device_passable = Self._device_passable
    comptime access_enabled = Self._access_enabled
    comptime TileFusion = Self._TileFusion
    comptime _shape_types = Self.LayoutType._shape_types
    comptime __shape_types = Self.LayoutType.__shape_types
    var _tile_fusion: Self._TileFusion
    var _layout: Self.LayoutType
    var handle: Pointer[Scalar[Self.dtype], UntrackedOrigin[mut = Self.mut]]

    comptime device_type: AnyType = _MoggLegacyTileTransformFusionTensor[
        Self._dtype,
        Self._LayoutType,
        Self._alignment,
        Self._mut,
        Self._TileFusion,
        False,
        True,
    ]

    comptime FusedViewType: TrivialRegisterPassable = Self.device_type

    @inline(.always)
    def __init__(
        out self,
        tile_fusion: Self._TileFusion,
        layout: Self.LayoutType,
        ptr: Pointer[Scalar[Self.dtype], UntrackedOrigin[mut = Self.mut]],
    ):
        self._tile_fusion = tile_fusion
        self._layout = layout
        self.handle = ptr

    @inline(.always)
    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ) where Self._device_passable:
        comptime fusion_offset = reflect[Self.device_type].field_offset[
            index=0, target=encoder.target()
        ]()
        encoder.encode_fields[Self._TileFusion](
            self._tile_fusion,
            target.unsafe_bitcast[UInt8]()
            .unsafe_offset(fusion_offset)
            .unsafe_bitcast[NoneType](),
        )
        comptime layout_offset = reflect[Self.device_type].field_offset[
            index=1, target=encoder.target()
        ]()
        encoder.encode(
            self._layout,
            target.unsafe_bitcast[UInt8]()
            .unsafe_offset(layout_offset)
            .unsafe_bitcast[NoneType](),
        )
        comptime handle_offset = reflect[Self.device_type].field_offset[
            index=2, target=encoder.target()
        ]()
        self.handle._to_device_type(
            encoder,
            target.unsafe_bitcast[UInt8]()
            .unsafe_offset(handle_offset)
            .unsafe_bitcast[NoneType](),
        )

    @staticmethod
    def get_type_name() -> String where Self._device_passable:
        return String("_MoggLegacyTileTransformFusionTensor")

    @inline(.always)
    def shape(self) -> Coord[*Self._shape_types]:
        return self._layout.shape_coord()

    @inline(.always)
    def layout(self) -> Self.LayoutType:
        return self._layout

    @inline(.always)
    def _to_tile_tensor(
        self,
    ) -> TileTensor[
        dtype = Self.dtype,
        LayoutType = Self.LayoutType,
        origin = UntrackedOrigin[mut = Self.mut],
    ]:
        return TileTensor[
            dtype = Self.dtype,
            LayoutType = Self.LayoutType,
            origin = UntrackedOrigin[mut = Self.mut],
        ](ptr=self.handle, layout=self._layout)

    @inline(.always)
    def tile_fusion(self) -> Self.TileFusion:
        return self._tile_fusion

    @inline(.always)
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
        comptime assert Self._access_enabled, (
            "a fused tile compute is only valid on the projected device value;"
            " the device-passable host proxy cannot compute directly"
        )
        return rebind[type_of(val)](
            self._tile_fusion.compute[dtype, rank, LayoutType, Copier](
                tile_coords, copier, val
            )
        )
)mojo";

} // namespace M::MOGG

#endif // GENERICML_GRAPHCOMPILER_MOGGDIALECT_SUPPORT_LEGACYTILEMOJOTRAITS_H
