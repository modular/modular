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
// Mojo source emitted once per module for the advanced-fusion MOGG-ABI lane:
// the buffer capture and the input-load / output-store / transform fusion
// argument definitions.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MOGGDIALECT_SUPPORT_MOJOTRAITS_H
#define GENERICML_GRAPHCOMPILER_MOGGDIALECT_SUPPORT_MOJOTRAITS_H

#include "llvm/ADT/StringRef.h"

namespace M::MOGG {

// Buffer identity (pointer + layout) at the region->stub boundary. The stub
// body builds the dense/fused wrappers from it and the contract metadata.
static constexpr llvm::StringLiteral kTensorBufferCaptureDefs = R"mojo(
from layout import TensorLayout


struct _MoggTensorBufferCapture[
    _dtype: DType,
    _LayoutType: TensorLayout,
](Copyable, Movable):
    comptime dtype = Self._dtype
    comptime LayoutType = Self._LayoutType
    var handle: Pointer[Scalar[Self.dtype], MutUntrackedOrigin]
    var _layout: Self._LayoutType

    @inline(.always)
    def __init__(
        out self,
        ptr: Pointer[Scalar[Self.dtype], MutUntrackedOrigin],
        layout: Self.LayoutType,
    ):
        self.handle = ptr
        self._layout = layout

    @inline(.always)
    def layout(self) -> Self.LayoutType:
        return self._layout
)mojo";

// The fused input-load argument (`Input & Fused & LoadAccess`):
//   * `InputLoadFusion` is the trait of the per-fusion functor emitted by
//     `emitInputLoadFusionFunctor`, which replays the producer body via dense
//     `tile.load(idx)`.
//   * `_MoggInputLoadFusionTensor` holds the functor and a layout and is
//     passed directly to `execute`; `load(idx)` delegates to the functor.
// One struct serves CPU and GPU. `_device_passable` lets the host proxy cross
// to the device; `_access_enabled` gates the callable surface, so GPU host
// proxies project through `device_type` to the access-enabled view. The
// functor captures bare `Pointer`s, which translate to device addresses; a
// `TileTensor` field would be bit-copied, so the tile is rebuilt in `__call__`.
//
// NOTE: this and `kOutputStoreFusionDefs` / `kTransformFusionDefs` are
// near-identical. If a fourth fused form appears, generate them from a
// builder instead of adding another verbatim block.
static constexpr llvm::StringLiteral kInputLoadFusionDefs = R"mojo(
from layout import Coord, TensorLayout, TileTensor
from extensibility import Fused, Input, LoadAccess, Loadable, Tensor, UnsafeFusedView
from std.builtin.device_passable import DevicePassable, DeviceTypeEncoder


trait InputLoadFusion(TrivialRegisterPassable):
    def __call__[
        dtype: DType,
        rank: Int,
        width: Int,
        element_alignment: Int = 1,
    ](self, idx: Coord) -> SIMD[dtype, width]:
        ...


# `LoadAccess` and `Loadable` are UNCONDITIONAL so a closure that captures the
# proxy can type-check a `.load()` call. The access gate is a `comptime assert
# Self._access_enabled` in the `load` body: `_access_enabled` is False on the
# device-passable HOST proxy and True on the projected `device_type` (device_type
# flips it), so a direct host-side load is a compile error while a load on the
# projected device value -- what the closure captures after projection -- is
# allowed. Same context gate as the store proxy.
struct _MoggInputLoadFusionTensor[
    _dtype: DType,
    _LayoutType: TensorLayout,
    _alignment: Int,
    _InFusion: InputLoadFusion,
    _device_passable: Bool,
    _access_enabled: Bool,
](
    Copyable,
    DevicePassable where _device_passable,
    Fused,
    Input,
    LoadAccess,
    Loadable,
    Movable,
    Tensor,
    TrivialRegisterPassable,
    UnsafeFusedView,
):
    comptime dtype = Self._dtype
    comptime LayoutType = Self._LayoutType
    comptime rank = Self.LayoutType.rank
    comptime alignment = Self._alignment
    comptime device_passable = Self._device_passable
    comptime access_enabled = Self._access_enabled
    comptime _shape_types = Self.LayoutType._shape_types
    comptime __shape_types = Self.LayoutType.__shape_types
    var in_fusion: Self._InFusion
    var _layout: Self.LayoutType

    comptime device_type: AnyType = _MoggInputLoadFusionTensor[
        Self._dtype,
        Self._LayoutType,
        Self._alignment,
        Self._InFusion,
        False,
        True,
    ]

    comptime FusedViewType: TrivialRegisterPassable = Self.device_type

    @inline(.always)
    def __init__(out self, in_fusion: Self._InFusion, layout: Self.LayoutType):
        self.in_fusion = in_fusion
        self._layout = layout

    @inline(.always)
    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ) where Self._device_passable:
        comptime in_fusion_offset = reflect[Self.device_type].field_offset[
            index=0, target=encoder.target()
        ]()
        encoder.encode_fields[Self._InFusion](
            self.in_fusion,
            target.unsafe_bitcast[UInt8]()
            .unsafe_offset(in_fusion_offset)
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
        return String("_MoggInputLoadFusionTensor")

    @inline(.always)
    def shape(self) -> Coord[*Self._shape_types]:
        return self._layout.shape_coord()

    @inline(.always)
    def load[
        width: Int,
        element_alignment: Int = 1,
    ](self, idx: Coord) -> SIMD[Self.dtype, width]:
        # Context gate (see the store proxy): `Loadable`/`LoadAccess` are
        # unconditional so a closure can type-check `.load()`; `_access_enabled`
        # is False on the device-passable host proxy and True on the projected
        # `device_type`, so a host-side load is a compile error while a load on
        # the projected device value is allowed.
        comptime assert Self._access_enabled, (
            "fused load is only valid on an access-enabled tensor (the projected"
            " device value); the device-passable host proxy cannot be loaded"
            " from directly"
        )
        return self.in_fusion.__call__[
            Self.dtype, Self.rank, width, element_alignment
        ](idx)
)mojo";

// The fused output-store argument (`Output & Fused & StoreAccess`), the
// epilogue counterpart of `kInputLoadFusionDefs`. `OutputStoreFusion.__call__`
// runs the fused consumer epilogue and writes. `Storable` is gated by
// `_access_enabled` as for fused load. The same wrapper fills a
// `_LegacyComputeOutput` slot whose epilogue can't be a compute lambda (a cast,
// reshape, or multi-store epilogue); `_compute_output_slot` adds that
// conformance there and only there, so a plain fused store never claims to be
// a compute-output slot.
static constexpr llvm::StringLiteral kOutputStoreFusionDefs = R"mojo(
from layout import Coord, TensorLayout, TileTensor
from extensibility import Fused, Output, StoreAccess, Storable, Tensor, UnsafeFusedView, _LegacyComputeOutput
from std.builtin.device_passable import DevicePassable, DeviceTypeEncoder


trait OutputStoreFusion(TrivialRegisterPassable):
    def __call__[
        dtype: DType,
        rank: Int,
        width: Int,
        element_alignment: Int = 1,
    ](self, idx: Coord, value: SIMD[dtype, width]):
        ...


# `StoreAccess` and `Storable` are UNCONDITIONAL, so a closure that captures the
# proxy can type-check a `.store()` call. The real access gate is a `comptime
# assert Self._access_enabled` in the `store` body: `_access_enabled` is False on
# the device-passable HOST proxy and True on the projected `device_type` (the
# device_type flips it). So a direct host-side store on the host proxy is a
# compile error, while a store on the projected device value -- which is what the
# closure body captures/runs on after projection -- is allowed. This is the
# context gate that denies host-side dereference of a device-passable proxy
# without any target-triple inspection.
struct _MoggOutputStoreFusionTensor[
    _dtype: DType,
    _LayoutType: TensorLayout,
    _alignment: Int,
    _OutFusion: OutputStoreFusion,
    _device_passable: Bool,
    _access_enabled: Bool,
    _compute_output_slot: Bool,
](
    Copyable,
    DevicePassable where _device_passable,
    Fused,
    Movable,
    Output,
    StoreAccess,
    Storable,
    Tensor,
    TrivialRegisterPassable,
    UnsafeFusedView,
    _LegacyComputeOutput where _compute_output_slot,
):
    comptime dtype = Self._dtype
    comptime LayoutType = Self._LayoutType
    comptime rank = Self.LayoutType.rank
    comptime alignment = Self._alignment
    comptime device_passable = Self._device_passable
    comptime access_enabled = Self._access_enabled
    comptime _shape_types = Self.LayoutType._shape_types
    comptime __shape_types = Self.LayoutType.__shape_types
    var out_fusion: Self._OutFusion
    var _layout: Self.LayoutType

    comptime device_type: AnyType = _MoggOutputStoreFusionTensor[
        Self._dtype,
        Self._LayoutType,
        Self._alignment,
        Self._OutFusion,
        False,
        True,
        Self._compute_output_slot,
    ]

    comptime FusedViewType: TrivialRegisterPassable = Self.device_type

    @inline(.always)
    def __init__(out self, out_fusion: Self._OutFusion, layout: Self.LayoutType):
        self.out_fusion = out_fusion
        self._layout = layout

    @inline(.always)
    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ) where Self._device_passable:
        comptime out_fusion_offset = reflect[Self.device_type].field_offset[
            index=0, target=encoder.target()
        ]()
        encoder.encode_fields[Self._OutFusion](
            self.out_fusion,
            target.unsafe_bitcast[UInt8]()
            .unsafe_offset(out_fusion_offset)
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
        return String("_MoggOutputStoreFusionTensor")

    @inline(.always)
    def shape(self) -> Coord[*Self._shape_types]:
        return self._layout.shape_coord()

    @inline(.always)
    def store[
        width: Int,
        element_alignment: Int = 1,
    ](self, idx: Coord, value: SIMD[Self.dtype, width]):
        # `Storable`/`StoreAccess` are unconditional so a host-created closure can
        # type-check `.store()`; the real gate is here in the body. `_access_enabled`
        # is False on the device-passable HOST proxy and True on the projected
        # `device_type`, so a direct host-side store is a compile error while a
        # store on the projected device value (the closure body runs there) is
        # allowed.
        comptime assert Self._access_enabled, (
            "fused store is only valid on an access-enabled tensor (the"
            " projected device value); the device-passable host proxy cannot be"
            " stored to directly"
        )
        self.out_fusion.__call__[
            Self.dtype, Self.rank, width, element_alignment
        ](idx, value)
)mojo";

// The fused output-transform (compute-output) argument. The kernel both
// transforms and writes one output edge, so a single operand composes
// `_LegacyComputable` (pure value->value) and `TileTensorable` (projects the
// output buffer for the parent's final store). The callable surfaces are gated
// by `_access_enabled`; the device kernel refines via `downcast[...,
// _LegacyComputable & TileTensorable]`. The buffer is a bare `Pointer` so it
// translates to a device address.
static constexpr llvm::StringLiteral kTransformFusionDefs = R"mojo(
from layout import Coord, TensorLayout, TileTensor
from extensibility import Fused, Output, TileTensorable, UnsafeFusedView, _LegacyComputable, _LegacyComputeOutput
from std.builtin.device_passable import DevicePassable, DeviceTypeEncoder


trait TransformFusion(TrivialRegisterPassable):
    def __call__[
        dtype: DType,
        rank: Int,
        width: Int,
        element_alignment: Int = 1,
    ](self, idx: Coord, value: SIMD[dtype, width]) -> SIMD[dtype, width]:
        ...


# `_LegacyComputeOutput`, `_LegacyComputable`, and `TileTensorable` are UNCONDITIONAL so a
# closure that captures the proxy can type-check `.compute()` / `.to_tile_tensor()`.
# The access gate is a `comptime assert Self._access_enabled` in each callable
# body: `_access_enabled` is False on the device-passable HOST proxy and True on
# the projected `device_type` (device_type flips it), so a direct host-side
# transform/store is a compile error while access on the projected device value --
# what the closure captures after projection -- is allowed. Same context gate as
# the store proxy.
struct _MoggTransformFusionTensor[
    _dtype: DType,
    _LayoutType: TensorLayout,
    _alignment: Int,
    _mut: Bool,
    _TransformFusion: TransformFusion,
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
    _LegacyComputable,
    _LegacyComputeOutput,
):
    comptime dtype = Self._dtype
    comptime LayoutType = Self._LayoutType
    comptime rank = Self.LayoutType.rank
    comptime alignment = Self._alignment
    comptime mut = Self._mut
    comptime device_passable = Self._device_passable
    comptime access_enabled = Self._access_enabled
    comptime _shape_types = Self.LayoutType._shape_types
    comptime __shape_types = Self.LayoutType.__shape_types
    var transform_fusion: Self._TransformFusion
    var _layout: Self.LayoutType
    var handle: Pointer[Scalar[Self.dtype], UntrackedOrigin[mut = Self.mut]]

    comptime device_type: AnyType = _MoggTransformFusionTensor[
        Self._dtype,
        Self._LayoutType,
        Self._alignment,
        Self._mut,
        Self._TransformFusion,
        False,
        True,
    ]

    comptime FusedViewType: TrivialRegisterPassable = Self.device_type

    @inline(.always)
    def __init__(
        out self,
        transform_fusion: Self._TransformFusion,
        layout: Self.LayoutType,
        ptr: Pointer[Scalar[Self.dtype], UntrackedOrigin[mut = Self.mut]],
    ):
        self.transform_fusion = transform_fusion
        self._layout = layout
        self.handle = ptr

    @inline(.always)
    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ) where Self._device_passable:
        comptime fusion_offset = reflect[Self.device_type].field_offset[
            index=0, target=encoder.target()
        ]()
        encoder.encode_fields[Self._TransformFusion](
            self.transform_fusion,
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
        return String("_MoggTransformFusionTensor")

    @inline(.always)
    def shape(self) -> Coord[*Self._shape_types]:
        return self._layout.shape_coord()

    @inline(.always)
    def layout(self) -> Self.LayoutType:
        return self._layout

    # `TileTensorable`: the parent kernel stores the transformed value through
    # this projection (rebuilt from the bare output-buffer pointer).
    @inline(.always)
    def _to_tile_tensor(
        self,
    ) -> TileTensor[
        dtype = Self.dtype,
        LayoutType = Self.LayoutType,
        origin = UntrackedOrigin[mut = Self.mut],
    ]:
        comptime assert Self._access_enabled, (
            "fused transform output projection is only valid on an access-enabled"
            " tensor (the projected device value); the device-passable host proxy"
            " cannot be projected to a tile directly"
        )
        return TileTensor[
            dtype = Self.dtype,
            LayoutType = Self.LayoutType,
            origin = UntrackedOrigin[mut = Self.mut],
        ](ptr=self.handle, layout=self._layout)

    @inline(.always)
    def compute[
        width: Int,
        element_alignment: Int = 1,
    ](self, idx: Coord, value: SIMD[Self.dtype, width]) -> SIMD[
        Self.dtype, width
    ]:
        comptime assert Self._access_enabled, (
            "fused transform is only valid on an access-enabled tensor (the"
            " projected device value); the device-passable host proxy cannot be"
            " transformed directly"
        )
        return self.transform_fusion.__call__[
            Self.dtype, Self.rank, width, element_alignment
        ](idx, value)
)mojo";

} // namespace M::MOGG

#endif // GENERICML_GRAPHCOMPILER_MOGGDIALECT_SUPPORT_MOJOTRAITS_H
