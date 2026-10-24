# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the MAX License:
# https://modular.com/legal/max-license
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
from std.builtin.device_passable import DevicePassable
from std.builtin.rebind import downcast
from std.collections import Array, Optional
from std.collections.string import StaticString
from std.math import ceildiv, iota
from std.ffi import external_call
from max.gpu import global_idx
from max.gpu.host.info import is_cpu
from std.sys.info import simd_width_of
from max.algorithm.functional import elementwise
import extensibility
from layout import TensorLayout
from layout import TileTensor, CoordLike, Coord, DefaultEngine, row_major
from layout.coord import DynamicCoord
from layout.tile_layout import Layout as TileLayout
from extensibility import StaticTensorSpec
from algorithm.reduce_op import ReduceOp
from max.gpu.host import DeviceContext, DeviceContextArray, Dim
from max.gpu.host import DeviceGraphBuilder
from extensibility import (
    InputTensor,
    InputVariadicTensors,
    ManagedTensorSlice,
    OutputTensor,
    OutputVariadicTensors,
    _FusedComputeOutputTensor,
    foreach,
    simd_load_from_managed_tensor_slice,
    simd_store_into_managed_tensor_slice,
)
from builtin_primitives.primitives import foreach_out_func
from extensibility import (
    _FusedInputTensor as FusedInputTensor,
)
from extensibility import (
    _FusedInputVariadicTensors as FusedInputVariadicTensors,
)
from extensibility import (
    _FusedOutputTensor as FusedOutputTensor,
)
from extensibility import (
    _FusedOutputVariadicTensors as FusedOutputVariadicTensors,
)
from extensibility import (
    _MutableInputTensor as MutableInputTensor,
)
from extensibility import (
    _TileProjection as TileProjection,
)
from extensibility import (
    StableAddrEngine,
    StableInputTensor,
    StableInputVariadicTensors,
)
from extensibility import (
    Fused,
    Input,
    LoadAccess,
    Loadable,
    MutableInput,
    Output,
    Stable,
    StoreAccess,
    Storable,
    Tensor,
    TensorTuple,
    TileTensorable,
    UnsafeFusedView,
    _LegacyComputable,
    _LegacyComputeOutput,
    unsafe_fused_view,
)

from std.utils import IndexList, StaticTuple
from std.utils.coord import Coord, coord_to_index_list


# TODO(MOCO-1413): remove this need to keep imported exported funcs alive.
@export
def export() abi("Mojo"):
    comptime _simd_load_from_managed_tensor_slice = simd_load_from_managed_tensor_slice
    comptime _simd_store_into_managed_tensor_slice = simd_store_into_managed_tensor_slice


# ===-----------------------------------------------------------------------===#
# Opaque Reg Types
# ===-----------------------------------------------------------------------===#


struct MyCustomScalarRegTrivialSI32(TrivialRegisterPassable):
    var val: Int32

    @implicit
    def __init__(out self, val: Int32):
        print("MyCustomScalarRegTrivialSI32.__init__", val)
        self.val = val


@extensibility.register("tensor_to_custom_scalar_si32_reg_trivial")
struct OpaqueToCustomScalarSI32RegTrivial:
    @staticmethod
    def execute(
        x: InputTensor[dtype=.int32, rank=1, ...]
    ) -> MyCustomScalarRegTrivialSI32:
        return MyCustomScalarRegTrivialSI32(x[0])


struct MyCustomScalarRegSI32(ImplicitlyCopyable, RegisterPassable):
    var val: Int32

    @implicit
    def __init__(out self, val: Int32):
        print("MyCustomScalarRegSI32.__init__", val)
        self.val = val

    def __deinit__(deinit self):
        print("MyCustomScalarRegSI32.__deinit__", self.val)


# It is intentional there are no methods which consume this.
# It is here to support some level of type checking.
struct MyCustomScalarRegF32(ImplicitlyCopyable, RegisterPassable):
    var val: Float32

    @implicit
    def __init__(out self, val: Float32):
        print("MyCustomScalarRegF32.__init__", val)
        self.val = val

    def __deinit__(deinit self):
        print("MyCustomScalarRegF32.__deinit__", self.val)


@extensibility.register("tensor_to_custom_scalar_si32_reg")
struct OpaqueToCustomScalarSI32Reg:
    @staticmethod
    def execute(
        x: InputTensor[dtype=.int32, rank=1, ...]
    ) -> MyCustomScalarRegSI32:
        return MyCustomScalarRegSI32(x[0])


# Adds two custom scalar types (one of which is register passable) and writes
# to a tensor
@extensibility.register("opaque_add_to_tensor_si32_reg")
struct OpaqueAddToTensorSI32Reg:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        outp: OutputTensor[dtype=.int32, rank=1, ...],
        x: MyCustomScalarRegSI32,
        y: MyCustomScalarRegSI32,
    ):
        outp[0] = x.val + y.val


@extensibility.register("opaque_add_to_tensor_f32_reg")
struct OpaqueAddToTensorF32:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        outp: OutputTensor[dtype=.float32, rank=1, ...],
        x: MyCustomScalarRegF32,
        y: MyCustomScalarRegF32,
    ):
        outp[0] = x.val + y.val


# ===-----------------------------------------------------------------------===#
# Opaque Mem. Types
# ===-----------------------------------------------------------------------===#


struct MyCustomScalarSI32(ImplicitlyCopyable):
    var val: Int32

    @implicit
    def __init__(out self, val: Int32):
        print("MyCustomScalarSI32.__init__", val)
        self.val = val

    def __deinit__(deinit self):
        print("MyCustomScalarSI32.__deinit__", self.val)


@extensibility.register("tensor_to_custom_scalar_si32")
struct OpaqueToCustomScalarSI32:
    @staticmethod
    def execute(
        x: InputTensor[dtype=.int32, rank=1, ...]
    ) -> MyCustomScalarSI32:
        return MyCustomScalarSI32(x[0])


# Adds two custom scalar types (one of which is register passable) and writes
# to a tensor
@extensibility.register("opaque_add_to_tensor_si32")
struct OpaqueAddToTensorSI32:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        outp: OutputTensor[dtype=.int32, rank=1, ...],
        x: MyCustomScalarSI32,
        y: MyCustomScalarSI32,
    ):
        outp[0] = x.val + y.val


@extensibility.register("opaque_add_to_tensor_si32_raises")
struct OpaqueAddToTensorSI32Raises:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        outp: OutputTensor[dtype=.int32, rank=1, ...],
        x: MyCustomScalarSI32,
        y: MyCustomScalarSI32,
    ) raises:
        outp[0] = x.val + y.val


# ===-----------------------------------------------------------------------===#
# Other Kernels
# ===-----------------------------------------------------------------------===#


@extensibility.register("imposter_add")
struct Foo:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        z: OutputTensor,
        x: InputTensor,
        y: InputTensor,
        ctx: DeviceContext,
    ) raises:
        @inline(.always)
        def func[width: Int](idx: Coord) {var} -> SIMD[z.dtype, width]:
            return rebind[SIMD[z.dtype, width]](x.load[width](idx)) + rebind[
                SIMD[z.dtype, width]
            ](y.load[width](idx))

        foreach[target=target](func, z, ctx)


@extensibility.register_shape_function("imposter_add")
def imposter_add_shape(
    x: Some[Tensor], y: Some[Tensor]
) -> IndexList[type_of(x).rank]:
    return rebind[IndexList[type_of(x).rank]](coord_to_index_list(x.shape()))


# Same as `imposter_add`, but its shape function raises, so the generated shape
# stub is a throwing one that also returns a value.
@extensibility.register("imposter_add_raising_shape")
struct FooRaisingShape:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        z: OutputTensor,
        x: InputTensor,
        y: InputTensor,
        ctx: DeviceContext,
    ) raises:
        @inline(.always)
        def func[width: Int](idx: Coord) {var} -> SIMD[z.dtype, width]:
            return rebind[SIMD[z.dtype, width]](x.load[width](idx)) + rebind[
                SIMD[z.dtype, width]
            ](y.load[width](idx))

        foreach[target=target](func, z, ctx)


@extensibility.register_shape_function("imposter_add_raising_shape")
def imposter_add_raising_shape_shape(
    x: Some[Tensor], y: Some[Tensor]
) raises -> IndexList[type_of(x).rank]:
    if type_of(x).rank == 0:
        raise "imposter_add_raising_shape: rank-0 is not supported"
    return rebind[IndexList[type_of(x).rank]](coord_to_index_list(x.shape()))


# Device entry for the GPU negate path: one thread per flat element, negating
# `input` into `output`. The trait arg's device representation is a `TileTensor`
# (see `_MoggDenseTensorArg.device_type`), so the kernel gets Coord-indexable
# tiles rather than bare pointers.
def _gpu_negate_kernel[
    dtype: DType, LayoutType: TensorLayout
](
    output: TileTensor[dtype, LayoutType, MutAnyOrigin],
    input: TileTensor[dtype, LayoutType, ImmutAnyOrigin],
    num_elements_dev: Int32,
):
    var num_elements = Int(num_elements_dev)
    var linear_idx = global_idx.x
    if linear_idx >= num_elements:
        return
    var output_flat = output.reshape(row_major(num_elements))
    var input_flat = input.reshape(row_major(num_elements))
    output_flat[Int(linear_idx)] = -input_flat[Int(linear_idx)]


# Trait-lane dense unary kernel (Slice 3 / Phase 4 e2e proof). Uses the
# composable tensor-argument traits instead of `ManagedTensorSlice`: the graph
# compiler emits an inline `_MoggDenseTensorArg` for each arg and infers the
# generic `T` from it. Computes `z[i] = -x[i]` over the flat element range.
# A single struct serves both targets: the CPU path runs the host
# `TileTensorable` projection; the GPU path crosses the same value as a
# `DevicePassable` and projects to a device `TileTensor`.
@extensibility.register("trait_negate")
struct TraitNegate:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        z: Some[Output & TileTensorable],
        x: Some[Input & TileTensorable],
        ctx: DeviceContext,
    ) raises:
        # The host-entry bound is the role + metadata traits; each target arm
        # asserts the capability it needs (mirrors `DenseElementwiseKernel`).
        comptime if is_cpu[target]():
            comptime assert conforms_to(type_of(z), TileTensorable)
            comptime assert conforms_to(type_of(x), TileTensorable)
            # `z.to_tile_tensor()` picks the mutable projection for an Output;
            # `x` gets the immutable one. `x` and `z` are independent generic
            # types, so `x` restates the output dtype to unify the two.
            var zt = z.to_tile_tensor()
            var xt = x.to_tile_tensor[type_of(z).dtype]()
            # The emitted dense arg does not override `Engine`, so its
            # projection resolves to the pointer-backed default. Flipping the
            # emitter to another engine must break here first.
            comptime assert type_of(zt).Engine == DefaultEngine[element_width=1]
            var n = Int(x.shape().product())
            var zf = zt.reshape(row_major(n))
            var xf = xt.reshape(row_major(n))
            for i in range(n):
                zf.store((i,), -xf.load((i,)))
        else:
            comptime assert conforms_to(type_of(z), DevicePassable)
            comptime assert conforms_to(type_of(x), DevicePassable)
            comptime block_size = 64
            var n = Int(x.shape().product())
            ctx.enqueue_function[
                _gpu_negate_kernel[type_of(z).dtype, type_of(z).LayoutType]
            ](
                z,
                x,
                Int32(n),
                grid_dim=(ceildiv(n, block_size),),
                block_dim=(block_size,),
            )


# Device entry for the stable negate path: `input` is a slot-backed tile
# (`StableAddrEngine`), so every load reads the data pointer out of the slot on
# the device before indexing.
def _gpu_stable_negate_kernel[
    dtype: DType, LayoutType: TensorLayout, in_origin: Origin
](
    output: TileTensor[dtype, LayoutType, MutAnyOrigin],
    input: TileTensor[
        dtype,
        LayoutType,
        in_origin,
        Engine=StableAddrEngine[element_width=1],
    ],
    num_elements_dev: Int32,
):
    var num_elements = Int(num_elements_dev)
    var linear_idx = global_idx.x
    if linear_idx >= num_elements:
        return
    var output_flat = output.reshape(row_major(num_elements))
    var input_flat = input.reshape(row_major(num_elements))
    output_flat[Int(linear_idx)] = -input_flat[Int(linear_idx)]


# Trait-lane `Stable` kernel: `x` arrives as an emitted `_MoggStableTensorArg`
# whose `Engine` is `StableAddrEngine`, so its projection keeps reading through
# the stable address slot. GPU only: the slot is device memory.
@extensibility.register("trait_stable_negate")
struct TraitStableNegate:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        z: Some[Output & TileTensorable],
        x: Some[Input & Tensor & Stable],
        ctx: DeviceContext,
    ) raises:
        comptime assert not is_cpu[
            target
        ](), "trait_stable_negate: a stable address slot is device memory"
        comptime assert conforms_to(type_of(z), DevicePassable)
        comptime assert conforms_to(type_of(x), TileTensorable)
        var xt = x.to_tile_tensor()
        # The stable arg must project through the slot engine; a pointer-backed
        # default here would read the slot address as data.
        comptime assert type_of(xt).Engine == StableAddrEngine[element_width=1]
        comptime block_size = 64
        var n = Int(x.shape().product())
        ctx.enqueue_function[
            _gpu_stable_negate_kernel[
                type_of(z).dtype, type_of(z).LayoutType, type_of(xt).origin
            ]
        ](
            z,
            xt,
            Int32(n),
            grid_dim=(ceildiv(n, block_size),),
            block_dim=(block_size,),
        )


# Top-level TileTensor projection variant of `trait_negate`: both CPU and GPU
# arms project dense trait args to `TileTensor` before dispatching.
@extensibility.register("trait_negate_tile_tensor")
struct TraitNegateTileTensor:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        z: Some[Output & TileTensorable],
        x: Some[Input & TileTensorable],
        ctx: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(z), TileTensorable)
        comptime assert conforms_to(type_of(x), TileTensorable)
        var zt = z.to_tile_tensor()
        var xt = x.to_tile_tensor[type_of(z).dtype]()
        comptime if is_cpu[target]():
            var n = Int(x.shape().product())
            var zf = zt.reshape(row_major(n))
            var xf = xt.reshape(row_major(n))
            for i in range(n):
                zf.store((i,), -xf.load((i,)))
        else:
            comptime block_size = 64
            var n = Int(x.shape().product())
            ctx.enqueue_function[
                _gpu_negate_kernel[type_of(zt).dtype, type_of(zt).LayoutType]
            ](
                zt,
                xt,
                Int32(n),
                grid_dim=(ceildiv(n, block_size),),
                block_dim=(block_size,),
            )


# UNIFIED-closure fused-load variant: the device kernel binds the PROJECTED load
# proxy (`downcast[XHost.device_type, Loadable]`, access-enabled) and reads it
# through a unified closure that captures that projected value. Same context gate
# as store: the closure captures `device_type` (created after projection), so the
# load body's `comptime assert _access_enabled` passes; a host-side load is a
# compile error (see `trait_fused_load_host_negative`).
def _gpu_fused_load_negate_unified_closure_kernel[
    XHost: DevicePassable,
    dtype: DType,
    LayoutType: TensorLayout,
](
    output: TileTensor[dtype, LayoutType, MutAnyOrigin],
    input: downcast[XHost.device_type, Loadable],
    n_dev: Int32,
):
    var n = Int(n_dev)
    var i = Int(global_idx.x)
    if i >= n:
        return

    @inline(.always)
    def load_fn(idx: Int) {imm input} -> SIMD[dtype, 1]:
        return rebind[SIMD[dtype, 1]](input.load[1]((idx,)))

    output.reshape(row_major(n))[i] = -load_fn(i)


@extensibility.register("trait_fused_load_negate_unified_closure")
struct TraitFusedLoadNegateUnifiedClosure:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        z: Some[Output & TileTensorable],
        x: Some[Input & Fused & LoadAccess],
        ctx: DeviceContext,
    ) raises:
        comptime if is_cpu[target]():
            comptime assert conforms_to(type_of(z), TileTensorable)
            comptime assert conforms_to(type_of(x), Loadable)
            var zt = z.to_tile_tensor()
            var n = Int(z.shape().product())
            var zf = zt.reshape(row_major(n))
            for i in range(n):
                var v = x.load[1]((i,))
                zf.store((i,), rebind[SIMD[type_of(zt).dtype, 1]](-v))
        else:
            comptime assert conforms_to(type_of(z), DevicePassable)
            comptime assert conforms_to(type_of(x), DevicePassable)
            comptime block_size = 64
            var n = Int(z.shape().product())
            ctx.enqueue_function[
                _gpu_fused_load_negate_unified_closure_kernel[
                    type_of(x), type_of(z).dtype, type_of(z).LayoutType
                ]
            ](
                z,
                x,
                Int32(n),
                grid_dim=(ceildiv(n, block_size),),
                block_dim=(block_size,),
            )


# NEGATIVE control: loads DIRECTLY from the fused-load proxy at the host entry.
# On a GPU device the emitted proxy is the device-passable HOST proxy
# (`_access_enabled=False`), so the load body's `comptime assert _access_enabled`
# fails at instantiation -- must NOT compile. Exercised (expecting failure) by
# `mogg-kernel-api-e2e-trait-fused-load-host-negative.mlir`.
@extensibility.register("trait_fused_load_host_negative")
struct TraitFusedLoadHostNegative:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        z: Some[Output & TileTensorable],
        x: Some[Input & Fused & LoadAccess],
        ctx: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(x), DevicePassable)
        comptime assert conforms_to(type_of(x), Loadable)
        # Direct host-side load on the device-passable proxy: the context gate
        # (`comptime assert _access_enabled` in the load body) rejects this.
        _ = x.load[1]((0,))


@inline(.always)
def _gpu_fused_load_negate_host_unified_closure_kernel[
    dtype: DType,
    LayoutType: TensorLayout,
    LoadFn: TrivialRegisterPassable
    & def[dtype: DType, width: Int](Int) -> SIMD[dtype, width],
](
    zt: TileTensor[dtype, LayoutType, MutUntrackedOrigin, Engine=_],
    n_dev: Int32,
    load_fn: LoadFn,
):
    var i = Int(global_idx.x)
    if i >= Int(n_dev):
        return
    zt.reshape(row_major(Int(n_dev))).store((i,), -load_fn[dtype, 1](i))


@inline(.always)
def _fused_load_negate[
    Z: Output & TileTensorable,
    //,
    target: StaticString,
](
    ctx: DeviceContext,
    z: Z,
    n: Int,
    load_fn: Some[def[dtype: DType, width: Int](Int) -> SIMD[dtype, width]],
) raises:
    """Dispatches a host-built load callback on CPU or GPU.

    `load_fn` is created in `execute` and threaded down as a value. On GPU it must
    be re-captured into a nullary device closure, because `load_fn` is not
    `DevicePassable` and cannot be a runtime argument to `enqueue_function`; a
    regular device call can still take it as a comptime parameter.
    """
    comptime assert conforms_to(Z, TileTensorable)
    comptime assert conforms_to(type_of(load_fn), TrivialRegisterPassable)
    comptime if is_cpu[target]():
        var zt = z.to_tile_tensor()
        var zf = zt.reshape(row_major(n))
        for i in range(n):
            zf.store((i,), -load_fn[type_of(zt).dtype, 1](i))
    else:
        var zt = z.to_tile_tensor()
        var n_dev = Int32(n)

        # Re-capture the load closure inside the kernel closure.
        def device_body() {var load_fn, var zt, var n_dev}:
            _gpu_fused_load_negate_host_unified_closure_kernel(
                zt, n_dev, load_fn
            )

        comptime block_size = 64
        ctx.enqueue_function(
            device_body,
            grid_dim=(ceildiv(n, block_size),),
            block_dim=(block_size,),
        )


# Host-built UNIFIED-closure fused-load variant. This is a distinct (and also
# real) pattern from the parameter-closure route: the closure is built on the
# host as a first-class value, threaded down through a helper, and either
# called directly on CPU or re-captured into the device-kernel body on GPU.
# It proves the projection works for kernels that layer this way (e.g. the
# CPU path of `RMSNormFusedResidualAdd`).
@extensibility.register("trait_fused_load_host_unified_closure")
struct TraitFusedLoadHostUnifiedClosure:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        z: Some[Output & TileTensorable],
        x: Some[Input & Fused & LoadAccess],
        ctx: DeviceContext,
    ) raises:
        var n = Int(z.shape().product())
        comptime if is_cpu[target]():
            # CPU: arg already access-enabled, capture directly.
            comptime assert conforms_to(
                type_of(x),
                Loadable & TrivialRegisterPassable,
            )

            @inline(.always)
            def cpu_load_fn[
                dtype: DType, width: Int
            ](idx: Int) {var x} -> SIMD[dtype, width]:
                return rebind[SIMD[dtype, width]](x.load[width]((idx,)))

            _fused_load_negate[target=target](ctx, z, n, cpu_load_fn)
        else:
            # GPU: project via accessor and capture the access-enabled view.
            comptime assert conforms_to(type_of(x), DevicePassable)
            comptime assert conforms_to(type_of(x), UnsafeFusedView)
            var xv = unsafe_fused_view(x)
            comptime assert conforms_to(type_of(xv), Loadable)

            @inline(.always)
            def gpu_load_fn[
                dtype: DType, width: Int
            ](idx: Int) {var xv} -> SIMD[dtype, width]:
                return rebind[SIMD[dtype, width]](xv.load[width]((idx,)))

            _fused_load_negate[target=target](ctx, z, n, gpu_load_fn)


# NEGATIVE control: a host-built closure that captures the fused-load host
# proxy directly must fail at launch because the closure body reaches the
# `_access_enabled = False` proxy and trips the load gate. Exercised (expecting
# failure) by `mogg-kernel-api-e2e-trait-fused-load-host-closure-negative.mlir`.
# Positive counterpart:
# `mogg-kernel-api-e2e-trait-fused-load-host-unified-closure.mlir`.
@extensibility.register("trait_fused_load_host_closure_negative")
struct TraitFusedLoadHostClosureNegative:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        z: Some[Output & TileTensorable],
        x: Some[Input & Fused & LoadAccess],
        ctx: DeviceContext,
    ) raises:
        var n = Int(z.shape().product())
        # Intentionally no `unsafe_fused_view(x)` projection: this kernel
        # captures `x` directly so the launched closure's body reaches the
        # `_access_enabled = False` proxy and the load gate fires -- the
        # failure this negative exists to exercise.
        comptime assert conforms_to(type_of(x), DevicePassable)

        # `conforms_to` refines the type in this scope, making `x`
        # `Loadable & ImplicitlyCopyable` so the closure captures the host
        # proxy itself -- the case this negative is meant to reject.
        comptime assert conforms_to(
            type_of(x), Loadable & ImplicitlyCopyable & TrivialRegisterPassable
        )

        # Not the failing line, despite appearances, and neither is the call
        # below binding it: closure `__call__` is instantiated lazily, so
        # defining this body and passing it under a closure trait bound both
        # compile on their own.
        def load_fn[
            dtype: DType, width: Int
        ](idx: Int) {var x} -> SIMD[dtype, width]:
            return rebind[SIMD[dtype, width]](x.load[width]((idx,)))

        # The failure fires inside this call, at the launch that re-captures
        # `load_fn`. The capturing endpoint compiles the closure body on the
        # host, passing the encoded device-type instance, so the body sees the
        # `_access_enabled = False` proxy and the load gate rejects it.
        _fused_load_negate[target=target](ctx, z, n, load_fn)


# Trait-lane in-place mutable kernel: `x` is read and written through the same
# buffer (`x[i] += 1`). Proves the emitted `_MoggDenseTensorArg[io=MUTABLE_INPUT]`
# satisfies a `MutableInput & TileTensorable` stub bound -- i.e.
# the wrapper conforms to `MutableInput` (gated on the io), not just `Input`.
@extensibility.register("trait_inplace_increment")
struct TraitInplaceIncrement:
    @staticmethod
    def execute[
        target: StaticString,
    ](x: Some[MutableInput & TileTensorable], ctx: DeviceContext) raises:
        comptime assert conforms_to(type_of(x), MutableInput)
        comptime assert conforms_to(type_of(x), TileTensorable)
        var xt = x.to_tile_tensor()
        var n = Int(x.shape().product())
        var xf = xt.reshape(row_major(n))
        for i in range(n):
            xf.store(
                (i,),
                rebind[SIMD[type_of(xt).dtype, 1]](xf.load((i,)) + 1),
            )


# ===-----------------------------------------------------------------------===#
# Trait-lane FUSED kernels (Phase 5): one per fusion access form.
#
# A fused operand carries the `Fused` marker plus one graph-visible access form:
#   * `Input & Fused & LoadAccess` -- prologue (input-load) fusion.
#     The callable `Loadable` view is used only where `x.load(idx)` is valid.
#   * `Output & Fused & StoreAccess` -- epilogue (output-store) fusion.
#     The callable `Storable` view is used only where `z.store(idx, v)` is valid.
#   * `Output & Fused & _LegacyComputable` -- epilogue (transform / compute-output)
#     fusion: `z.compute(idx, v)` runs the consumer and returns the value;
#     the GC output mechanism performs the final dense store.
# For each, the GC emits an inline `_MoggInput*/Output*FusionArg` wrapper around
# the existing fusion functor (`InputFusion`/`OutputFusion`/`ComputeOutputFusion`)
# conforming to the landed access trait; the kernel infers its generic `T`.
# Element access is `Coord`-based (the trait interface); the wrapper bridges to
# the functor's `IndexList`-based method.
# ===-----------------------------------------------------------------------===#


# Input-load (prologue) fusion: z[i] = -x.load(i), rank-1. Mirrors
# `trait_negate` but with a fused input instead of a dense one.
# Device entry for the fused-load negate path: one thread per element negates the
# producer's fused load into the dense output. `input` is the device-projected
# fused-load wrapper, bound by ONLY its `Loadable` capability via
# `downcast[XHost.device_type, Loadable]` (the `XHost: DevicePassable` generic is
# just the carrier that names the device type).
def _gpu_fused_load_negate_kernel[
    XHost: DevicePassable,
    dtype: DType,
    LayoutType: TensorLayout,
](
    output: TileTensor[dtype, LayoutType, MutAnyOrigin],
    input: downcast[XHost.device_type, Loadable],
    n_dev: Int32,
):
    var n = Int(n_dev)
    var i = Int(global_idx.x)
    if i >= n:
        return
    output.reshape(row_major(n))[i] = rebind[SIMD[dtype, 1]](
        -input.load[1]((i,))
    )


@extensibility.register("trait_fused_load_negate")
struct TraitFusedLoadNegate:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        z: Some[Output & TileTensorable],
        x: Some[Input & Fused & LoadAccess],
        ctx: DeviceContext,
    ) raises:
        comptime if is_cpu[target]():
            comptime assert conforms_to(type_of(z), TileTensorable)
            comptime assert conforms_to(type_of(x), Loadable)
            var zt = z.to_tile_tensor()
            var n = Int(z.shape().product())
            var zf = zt.reshape(row_major(n))
            for i in range(n):
                var v = x.load[1]((i,))
                zf.store((i,), rebind[SIMD[type_of(zt).dtype, 1]](-v))
        else:
            comptime assert conforms_to(type_of(z), DevicePassable)
            comptime assert conforms_to(type_of(x), DevicePassable)
            comptime block_size = 64
            var n = Int(z.shape().product())
            ctx.enqueue_function[
                _gpu_fused_load_negate_kernel[
                    type_of(x), type_of(z).dtype, type_of(z).LayoutType
                ]
            ](
                z,
                x,
                Int32(n),
                grid_dim=(ceildiv(n, block_size),),
                block_dim=(block_size,),
            )


# Device entry for `trait_tensor_tuple_add`: one thread per element of the
# concatenated pairs, which finds the pair `k` that owns it. Both tuples are
# spelled as the host tuples' `device_type`, and element `k` is typed only by
# `AnyType` in the body, so it is rebound to its `TileProjection`, mutable for
# an output and immutable for an input. The element types are `TypeList`
# parameters because a signature takes one list per tuple.
# TODO(GEX-4386): Type element `k` without the `TileProjection` rebind.
def _gpu_tensor_tuple_add_kernel[
    Outs: TypeList[Trait=Output & TileTensorable, ...],
    Ins: TypeList[Trait=Input & TileTensorable, ...],
    bias_dtype: DType,
    BiasLayoutType: TensorLayout,
](
    outputs: TensorTuple[*Outs].device_type,
    bias: TileTensor[bias_dtype, BiasLayoutType, ImmutAnyOrigin],
    inputs: TensorTuple[*Ins].device_type,
    offsets: StaticTuple[Int32, len(Ins) + 1],
    n_dev: Int32,
):
    var i = Int32(global_idx.x)
    if i >= n_dev:
        return
    var b = bias.reshape(row_major(bias.num_elements()))[0]
    comptime for k in range(len(Ins)):
        if i >= offsets[k] and i < offsets[k + 1]:
            comptime out_dtype = Outs[k].dtype
            var z = rebind[TileProjection[True, Outs[k]]](outputs[k])
            var x = rebind[TileProjection[False, Ins[k]]](inputs[k])
            var j = Int(i - offsets[k])
            var n = Int(offsets[k + 1] - offsets[k])
            var v = x.reshape(row_major(n)).load[width=1]((j,))
            z.reshape(row_major(n)).store(
                (j,), v.cast[out_dtype]() + b.cast[out_dtype]()
            )


# `TensorTuple` kernel with a variadic on each side:
# `outputs[i] = inputs[i] + bias[0]`, cast to the output dtype. Each element is
# its own emitted `_MoggDenseTensorArg`, so the elements may differ in dtype and
# length. `bias` sits between the two tuples. On the GPU both tuples cross the
# launch as `DevicePassable`s, so the output tuple's elements arrive as mutable
# device `TileTensor`s.
@extensibility.register("trait_tensor_tuple_add")
struct TraitTensorTupleAdd:
    @staticmethod
    def execute[
        Outs: TypeList[Trait=Output & TileTensorable, ...],
        Ins: TypeList[Trait=Input & TileTensorable, ...],
        //,
        target: StaticString,
    ](
        outputs: TensorTuple[*Outs],
        bias: Some[Input & TileTensorable],
        inputs: TensorTuple[*Ins],
        ctx: DeviceContext,
    ) raises:
        comptime if is_cpu[target]():
            comptime assert len(Outs) == len(Ins)
            comptime assert conforms_to(type_of(bias), TileTensorable)
            var bt = bias.to_tile_tensor()
            comptime for i in range(len(Ins)):
                comptime assert conforms_to(type_of(outputs[i]), TileTensorable)
                comptime assert conforms_to(type_of(inputs[i]), TileTensorable)
                var zt = outputs[i].to_tile_tensor()
                var xt = inputs[i].to_tile_tensor()
                comptime OutT = SIMD[type_of(zt).dtype, 1]
                var b = (
                    bt.reshape(row_major(bt.num_elements()))
                    .load[width=1]((0,))
                    .cast[type_of(zt).dtype]()
                )
                var n = Int(inputs[i].shape().product())
                var zf = zt.reshape(row_major(n))
                var xf = xt.reshape(row_major(n))
                for j in range(n):
                    var v = xf.load[width=1]((j,))
                    zf.store(
                        (j,), rebind[OutT](v.cast[type_of(zt).dtype]() + b)
                    )
        else:
            comptime assert len(Outs) == len(Ins)
            comptime assert conforms_to(type_of(outputs), DevicePassable)
            comptime assert conforms_to(type_of(inputs), DevicePassable)
            comptime assert conforms_to(type_of(bias), DevicePassable)
            var offsets = StaticTuple[Int32, len(Ins) + 1](fill=0)
            comptime for k in range(len(Ins)):
                offsets[k + 1] = offsets[k] + Int32(
                    Int(inputs[k].shape().product())
                )
            var n = Int(offsets[len(Ins)])
            comptime block_size = 64
            ctx.enqueue_function[
                _gpu_tensor_tuple_add_kernel[
                    Outs, Ins, type_of(bias).dtype, type_of(bias).LayoutType
                ]
            ](
                outputs,
                bias,
                inputs,
                offsets,
                Int32(n),
                grid_dim=(ceildiv(n, block_size),),
                block_dim=(block_size,),
            )


# Device entry for `trait_tensor_tuple_add_array`: one thread per element of the
# concatenated pairs, which finds the pair `k` that owns it. Both tuples arrive
# as `Array`s of `TileTensor`s built on the host.
def _gpu_array_tuple_add_kernel[
    out_dtype: DType,
    OutputLayoutType: TensorLayout,
    bias_dtype: DType,
    BiasLayoutType: TensorLayout,
    in_dtype: DType,
    InputLayoutType: TensorLayout,
    size: Int,
](
    outputs: Array[TileTensor[out_dtype, OutputLayoutType, MutAnyOrigin], size],
    bias: TileTensor[bias_dtype, BiasLayoutType, ImmutAnyOrigin],
    inputs: Array[TileTensor[in_dtype, InputLayoutType, ImmutAnyOrigin], size],
    n_dev: Int32,
):
    var i = Int(global_idx.x)
    if i >= Int(n_dev):
        return
    var b = bias.reshape(row_major(bias.num_elements()))[0].cast[out_dtype]()
    var offset = 0
    comptime for k in range(size):
        var n = inputs[k].num_elements()
        if i < offset + n:
            var j = i - offset
            var v = inputs[k][j]
            outputs[k][j] = v.cast[out_dtype]() + b
            return
        offset += n


# `trait_tensor_tuple_add` with each tuple converted to an `Array` of
# `TileTensor`s on the host, as `mo.split` does. Each side's elements share one
# type, so they must share one dtype, and their layouts become dynamic. GPU
# only.
@extensibility.register("trait_tensor_tuple_add_array")
struct TraitTensorTupleAddArray:
    @staticmethod
    def execute[
        Outs: TypeList[Trait=Output & TileTensorable, ...],
        Ins: TypeList[Trait=Input & TileTensorable, ...],
        //,
        target: StaticString,
    ](
        outputs: TensorTuple[*Outs],
        bias: Some[Input & TileTensorable],
        inputs: TensorTuple[*Ins],
        ctx: DeviceContext,
    ) raises:
        comptime assert len(Outs) == len(Ins)
        comptime assert conforms_to(type_of(bias), TileTensorable)
        var bt = bias.to_tile_tensor()
        comptime if is_cpu[target]():
            raise Error("trait_tensor_tuple_add_array only runs on the GPU")
        else:
            comptime out_dtype = Outs[0].dtype
            comptime in_dtype = Ins[0].dtype
            comptime DynamicLayout1D = TileLayout[
                shape_types=DynamicCoord[.int64, 1].element_types,
                stride_types=DynamicCoord[.int64, 1].element_types,
            ]
            comptime OutputTile = TileTensor[
                out_dtype, DynamicLayout1D, MutAnyOrigin
            ]
            comptime InputTile = TileTensor[
                in_dtype, DynamicLayout1D, ImmutAnyOrigin
            ]

            def output_tile[k: Int]() {imm outputs} -> OutputTile:
                comptime assert (
                    Outs[k].dtype == out_dtype
                ), "trait_tensor_tuple_add_array's outputs must share one dtype"
                comptime assert conforms_to(type_of(outputs[k]), TileTensorable)
                var zt = outputs[k].to_tile_tensor()
                return rebind[OutputTile](zt.make_dynamic[.int64]())

            def input_tile[k: Int]() {imm inputs} -> InputTile:
                comptime assert (
                    Ins[k].dtype == in_dtype
                ), "trait_tensor_tuple_add_array's inputs must share one dtype"
                comptime assert conforms_to(type_of(inputs[k]), TileTensorable)
                var xt = inputs[k].to_tile_tensor()
                return rebind[InputTile](xt.make_dynamic[.int64]())

            var output_tiles = Array[OutputTile, len(Outs)](
                fill_with_unrolled=output_tile
            )
            var input_tiles = Array[InputTile, len(Ins)](
                fill_with_unrolled=input_tile
            )
            var n = 0
            comptime for k in range(len(Ins)):
                n += input_tiles[k].num_elements()
            comptime block_size = 64
            ctx.enqueue_function[
                _gpu_array_tuple_add_kernel[
                    out_dtype,
                    DynamicLayout1D,
                    type_of(bt).dtype,
                    type_of(bt).LayoutType,
                    in_dtype,
                    DynamicLayout1D,
                    len(Ins),
                ]
            ](
                output_tiles,
                bt,
                input_tiles,
                Int32(n),
                grid_dim=(ceildiv(n, block_size),),
                block_dim=(block_size,),
            )


# Device entry for `trait_tensor_tuple_fused_concat`: one thread per output
# element, which finds the tuple element `k` that owns it and loads through that
# element's fused producer. `inputs` is the host tuple's `device_type`, so
# element `k` is the access-enabled view of `Ins[k]`, refined to `Loadable` at
# the load.
def _gpu_fused_tuple_concat_kernel[
    dtype: DType,
    LayoutType: TensorLayout,
    *Ins: LoadAccess,
](
    output: TileTensor[dtype, LayoutType, MutAnyOrigin],
    inputs: TensorTuple[*Ins].device_type,
    offsets: StaticTuple[Int32, len(Ins) + 1],
    n_dev: Int32,
):
    var i = Int32(global_idx.x)
    if i >= n_dev:
        return
    comptime for k in range(len(Ins)):
        if i >= offsets[k] and i < offsets[k + 1]:
            comptime assert conforms_to(type_of(inputs[k]), Loadable)
            output.reshape(row_major(Int(n_dev)))[Int(i)] = (
                inputs[k].load[1]((Int(i - offsets[k]),)).cast[dtype]()
            )


# `mo.concat`'s shape with a fused `TensorTuple`: concatenates rank-1 inputs
# into `z`, casting each to the output dtype. Each element is the fused
# producer's `_MoggInputLoadFusionTensor`, so the elements may differ in dtype
# and length. Like `Concat`, the CPU arm captures the whole input set in one
# closure and loads element `k` by comptime index; the GPU arm launches with
# the tuple itself.
@extensibility.register("trait_tensor_tuple_fused_concat")
struct TraitTensorTupleFusedConcat:
    @staticmethod
    def execute[
        Ins: TypeList[Trait=Input & Fused & LoadAccess, ...],
        //,
        target: StaticString,
    ](
        z: Some[Output & TileTensorable],
        inputs: TensorTuple[*Ins],
        ctx: DeviceContext,
    ) raises:
        comptime if is_cpu[target]():
            comptime assert conforms_to(type_of(z), TileTensorable)
            var zt = z.to_tile_tensor()
            comptime out_dtype = type_of(zt).dtype

            @inline(.always)
            def load_input[k: Int](j: Int) {imm inputs} -> Scalar[out_dtype]:
                comptime assert conforms_to(type_of(inputs[k]), Loadable)
                return inputs[k].load[1]((j,)).cast[out_dtype]()

            var zf = zt.reshape(row_major(Int(z.shape().product())))
            var offset = 0
            comptime for k in range(len(Ins)):
                var n = Int(inputs[k].shape().product())
                for j in range(n):
                    zf.store((offset + j,), load_input[k](j))
                offset += n
        else:
            # The tuple holds host-side proxies, so it crosses as a
            # `DevicePassable`, like `_MoggDenseTensorArg`: the launch
            # projects each element to its access-enabled device view.
            comptime assert conforms_to(type_of(z), TileTensorable)
            comptime assert conforms_to(type_of(inputs), DevicePassable)
            var zt = z.to_tile_tensor()
            var offsets = StaticTuple[Int32, len(Ins) + 1](fill=0)
            comptime for k in range(len(Ins)):
                offsets[k + 1] = offsets[k] + Int32(
                    Int(inputs[k].shape().product())
                )
            var n = Int(offsets[len(Ins)])
            comptime block_size = 64
            ctx.enqueue_function[
                _gpu_fused_tuple_concat_kernel[
                    type_of(zt).dtype, type_of(zt).LayoutType, *Ins
                ]
            ](
                zt,
                inputs,
                offsets,
                Int32(n),
                grid_dim=(ceildiv(n, block_size),),
                block_dim=(block_size,),
            )


# Device entry for `trait_tensor_tuple_fused_store_split`: one thread per input
# element, which finds the tuple element `k` that owns it and stores through
# that element's fused consumer. `outputs` is the host tuple's `device_type`, so
# element `k` is the access-enabled view of `Outs[k]`, refined to `Storable` at
# the store.
def _gpu_fused_tuple_split_kernel[
    dtype: DType,
    LayoutType: TensorLayout,
    *Outs: StoreAccess,
](
    outputs: TensorTuple[*Outs].device_type,
    input: TileTensor[dtype, LayoutType, ImmutAnyOrigin],
    offsets: StaticTuple[Int32, len(Outs) + 1],
    n_dev: Int32,
):
    var i = Int32(global_idx.x)
    if i >= n_dev:
        return
    var v = input.reshape(row_major(Int(n_dev)))[Int(i)]
    comptime for k in range(len(Outs)):
        if i >= offsets[k] and i < offsets[k + 1]:
            comptime assert conforms_to(type_of(outputs[k]), Storable)
            outputs[k].store[1](
                (Int(i - offsets[k]),),
                v.cast[type_of(outputs[k]).dtype](),
            )


# `mo.split`'s shape with a fused-output `TensorTuple`, the inverse of
# `trait_tensor_tuple_fused_concat`: splits rank-1 `x` across the outputs,
# casting each run to its output's dtype. Each element is the fused consumer's
# `_MoggOutputStoreFusionTensor`, so a store runs that output's epilogue.
@extensibility.register("trait_tensor_tuple_fused_store_split")
struct TraitTensorTupleFusedStoreSplit:
    @staticmethod
    def execute[
        Outs: TypeList[Trait=Output & Fused & StoreAccess, ...],
        //,
        target: StaticString,
    ](
        outputs: TensorTuple[*Outs],
        x: Some[Input & TileTensorable],
        ctx: DeviceContext,
    ) raises:
        comptime if is_cpu[target]():
            comptime assert conforms_to(type_of(x), TileTensorable)
            var xt = x.to_tile_tensor()
            var xf = xt.reshape(row_major(Int(x.shape().product())))
            var offset = 0
            comptime for k in range(len(Outs)):
                comptime assert conforms_to(type_of(outputs[k]), Storable)
                var n = Int(outputs[k].shape().product())
                for j in range(n):
                    var v = xf.load[width=1]((offset + j,))
                    outputs[k].store[1](
                        (j,), v.cast[type_of(outputs[k]).dtype]()
                    )
                offset += n
        else:
            comptime assert conforms_to(type_of(outputs), DevicePassable)
            comptime assert conforms_to(type_of(x), DevicePassable)
            var offsets = StaticTuple[Int32, len(Outs) + 1](fill=0)
            comptime for k in range(len(Outs)):
                offsets[k + 1] = offsets[k] + Int32(
                    Int(outputs[k].shape().product())
                )
            var n = Int(offsets[len(Outs)])
            comptime block_size = 64
            ctx.enqueue_function[
                _gpu_fused_tuple_split_kernel[
                    type_of(x).dtype, type_of(x).LayoutType, *Outs
                ]
            ](
                outputs,
                x,
                offsets,
                Int32(n),
                grid_dim=(ceildiv(n, block_size),),
                block_dim=(block_size,),
            )


# Output-store (epilogue) fusion: computes `2 * x[i]` and hands it to the fused
# store, which runs the consumer epilogue and writes. `x` is a plain dense input.
# Device entry for the fused-store double path: one thread per element computes
# `2 * x` and hands it to the device-projected fused-store wrapper, bound by ONLY
# its `Storable` capability via `downcast[ZHost.device_type, Storable]`.
def _gpu_fused_store_double_kernel[
    ZHost: DevicePassable,
    dtype: DType,
    LayoutType: TensorLayout,
](
    output: downcast[ZHost.device_type, Storable],
    input: TileTensor[dtype, LayoutType, ImmutAnyOrigin],
    n_dev: Int32,
):
    var n = Int(n_dev)
    var i = Int(global_idx.x)
    if i >= n:
        return
    var v = rebind[SIMD[dtype, 1]](input.reshape(row_major(n))[i]) * 2
    output.store[1]((i,), rebind[SIMD[type_of(output).dtype, 1]](v))


@extensibility.register("trait_fused_store_double")
struct TraitFusedStoreDouble:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        z: Some[Output & Fused & StoreAccess],
        x: Some[Input & TileTensorable],
        ctx: DeviceContext,
    ) raises:
        comptime if is_cpu[target]():
            comptime assert conforms_to(type_of(z), Storable)
            comptime assert conforms_to(type_of(x), TileTensorable)
            var xt = x.to_tile_tensor[type_of(z).dtype]()
            var n = Int(z.shape().product())
            var xf = xt.reshape(row_major(n))
            for i in range(n):
                var v = xf.load[width=1]((i,)) * 2
                z.store[1]((i,), v)
        else:
            comptime assert conforms_to(type_of(z), DevicePassable)
            comptime assert conforms_to(type_of(x), DevicePassable)
            comptime block_size = 64
            var n = Int(z.shape().product())
            ctx.enqueue_function[
                _gpu_fused_store_double_kernel[
                    type_of(z), type_of(z).dtype, type_of(x).LayoutType
                ]
            ](
                z,
                x,
                Int32(n),
                grid_dim=(ceildiv(n, block_size),),
                block_dim=(block_size,),
            )


def _gpu_fused_store_parameter_closure_kernel[
    dtype: DType,
    ZDev: Storable,
    LayoutType: TensorLayout,
    store_adapter: def[OutT: Storable, width: Int](
        OutT, Coord, SIMD[OutT.dtype, width]
    ) thin -> None,
](
    output: ZDev,
    input: TileTensor[dtype, LayoutType, ImmutAnyOrigin],
    n_dev: Int32,
):
    var n = Int(n_dev)
    var i = Int(global_idx.x)
    if i >= n:
        return
    store_adapter[ZDev, 1](
        output,
        Coord(i),
        rebind[SIMD[ZDev.dtype, 1]](input.reshape(row_major(n))[i]),
    )


@inline(.always)
def _launch_fused_store_parameter_closure[
    dtype: DType,
    ZHost: DevicePassable,
    XHost: DevicePassable,
    LayoutType: TensorLayout,
    store_adapter: def[OutT: Storable, width: Int](
        OutT, Coord, SIMD[OutT.dtype, width]
    ) thin -> None,
](output: ZHost, input: XHost, n: Int, ctx: DeviceContext,) raises:
    comptime block_size = 64
    ctx.enqueue_function[
        _gpu_fused_store_parameter_closure_kernel[
            dtype,
            downcast[ZHost.device_type, Storable],
            LayoutType,
            store_adapter,
        ]
    ](
        output,
        input,
        Int32(n),
        grid_dim=(ceildiv(n, block_size),),
        block_dim=(block_size,),
    )


@extensibility.register("trait_fused_store_parameter_closure")
struct TraitFusedStoreParameterClosure:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        z: Some[Output & Fused & StoreAccess],
        x: Some[Input & TileTensorable],
        ctx: DeviceContext,
    ) raises:
        comptime if is_cpu[target]():
            comptime assert conforms_to(type_of(z), Storable)
            comptime assert conforms_to(type_of(x), TileTensorable)
            var xt = x.to_tile_tensor[type_of(z).dtype]()
            var n = Int(z.shape().product())
            var xf = xt.reshape(row_major(n))
            for i in range(n):
                z.store[1]((i,), xf.load[width=1]((i,)))
        else:
            comptime assert conforms_to(type_of(z), DevicePassable)
            comptime assert conforms_to(type_of(x), DevicePassable)
            comptime dtype = type_of(x).dtype

            @inline(.always)
            def store_adapter[
                OutT: Storable, width: Int
            ](output: OutT, coord: Coord, value: SIMD[OutT.dtype, width]):
                output.store[width](coord, value)

            _launch_fused_store_parameter_closure[
                dtype,
                type_of(z),
                type_of(x),
                type_of(x).LayoutType,
                store_adapter,
            ](z, x, Int(z.shape().product()), ctx)


# Device entry for the UNIFIED-closure fused-store path. The device kernel binds
# the PROJECTED store proxy by its `Storable` capability via
# `downcast[ZHost.device_type, Storable]` -- the projected value has
# `_access_enabled=True`. It then creates a unified closure that captures that
# projected value and performs the store. This is the supported closure shape: the
# closure is created AFTER projection, so it captures `device_type` and the store
# body's `comptime assert _access_enabled` gate passes. A host-created closure
# over the device-passable HOST proxy (or a direct host-side store) hits
# `_access_enabled=False` and fails to compile -- see the negative test kernel.
def _gpu_fused_store_unified_closure_kernel[
    ZHost: DevicePassable,
    dtype: DType,
    LayoutType: TensorLayout,
](
    output: downcast[ZHost.device_type, Storable],
    input: TileTensor[dtype, LayoutType, ImmutAnyOrigin],
    n_dev: Int32,
):
    var n = Int(n_dev)
    var i = Int(global_idx.x)
    if i >= n:
        return
    var v = rebind[SIMD[dtype, 1]](input.reshape(row_major(n))[i]) * 2

    # Unified closure capturing the projected (access-enabled) store proxy.
    @inline(.always)
    def store_fn(idx: Int, val: SIMD[dtype, 1]) {imm output}:
        output.store[1]((idx,), rebind[SIMD[type_of(output).dtype, 1]](val))

    store_fn(i, v)


@extensibility.register("trait_fused_store_unified_closure")
struct TraitFusedStoreUnifiedClosure:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        z: Some[Output & Fused & StoreAccess],
        x: Some[Input & TileTensorable],
        ctx: DeviceContext,
    ) raises:
        comptime if is_cpu[target]():
            comptime assert conforms_to(type_of(z), Storable)
            comptime assert conforms_to(type_of(x), TileTensorable)
            var xt = x.to_tile_tensor[type_of(z).dtype]()
            var n = Int(z.shape().product())
            var xf = xt.reshape(row_major(n))
            for i in range(n):
                z.store[1]((i,), xf.load[width=1]((i,)) * 2)
        else:
            comptime assert conforms_to(type_of(z), DevicePassable)
            comptime assert conforms_to(type_of(x), DevicePassable)
            comptime block_size = 64
            var n = Int(z.shape().product())
            ctx.enqueue_function[
                _gpu_fused_store_unified_closure_kernel[
                    type_of(z), type_of(z).dtype, type_of(x).LayoutType
                ]
            ](
                z,
                x,
                Int32(n),
                grid_dim=(ceildiv(n, block_size),),
                block_dim=(block_size,),
            )


# NEGATIVE control: stores DIRECTLY on the fused-store proxy from the host entry,
# with no projection. On a GPU device the emitted proxy is the device-passable
# HOST proxy (`_access_enabled=False`), so the store body's
# `comptime assert Self._access_enabled` fails at instantiation -- this kernel must
# NOT compile through the gc-pipeline. Exercised (expecting failure) by
# `mogg-kernel-api-e2e-trait-fused-store-host-negative.mlir`.
@extensibility.register("trait_fused_store_host_negative")
struct TraitFusedStoreHostNegative:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        z: Some[Output & Fused & StoreAccess],
        x: Some[Input & TileTensorable],
        ctx: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(z), DevicePassable)
        comptime assert conforms_to(type_of(z), Storable)
        # Direct host-side store on the device-passable proxy: `store` resolves
        # (Storable is unconditional) but the store body's context gate
        # (`comptime assert _access_enabled`) rejects it at instantiation, because
        # the emitted host proxy has `access_enabled = False`.
        z.store[1]((0,), SIMD[type_of(z).dtype, 1](0))


# Output-transform (compute-output) fusion: the kernel computes `x[i] + 1`.
# In the compute form it applies the fused transform via `z.compute`, then
# performs the final dense store itself (a compute-form operand transforms a
# computed value and leaves the parent responsible for the store).
# `_LegacyComputable` is the sole access form; `TileTensorable` adds only the
# dense projection used for that final store. In the store form (the graph
# compiler downgraded a cast, reshape, or multi-store epilogue) it calls
# `z.store`, which runs the whole epilogue. `x` is a plain dense input.
# Device entry for the fused-transform path: one thread per element computes
# `x[i] + 1`, applies the fused transform via `z.compute`, and stores through
# the output-buffer projection `z.to_tile_tensor()`. `z` is the device-projected
# transform wrapper, bound by ONLY `_LegacyComputable & TileTensorable` via
# `downcast[ZHost.device_type, _LegacyComputable & TileTensorable]`.
def _gpu_fused_transform_bias_kernel[
    ZHost: DevicePassable,
    dtype: DType,
    LayoutType: TensorLayout,
](
    z: downcast[ZHost.device_type, _LegacyComputable & TileTensorable],
    x: TileTensor[dtype, LayoutType, ImmutAnyOrigin],
    n_dev: Int32,
):
    var n = Int(n_dev)
    var i = Int(global_idx.x)
    if i >= n:
        return
    var coord = Coord(i)
    var raw = rebind[SIMD[type_of(z).dtype, 1]](x.reshape(row_major(n))[i] + 1)
    var t = z.compute[1](coord, raw)
    # The device-side `z` carries no role, so ask for the writable view; an
    # Output edge is always mutable, which the projection checks.
    var zt = z.to_tile_tensor[mut=True]()
    zt.store(coord, rebind[SIMD[type_of(zt).dtype, 1]](t))


# Device entry for the store form of a `_LegacyComputeOutput` slot: one thread
# per element computes `x[i] + 1` and stores it through `z.store`, which runs
# the whole downgraded epilogue.
def _gpu_fused_transform_bias_store_form_kernel[
    ZHost: DevicePassable,
    dtype: DType,
    LayoutType: TensorLayout,
](
    z: downcast[ZHost.device_type, Storable],
    x: TileTensor[dtype, LayoutType, ImmutAnyOrigin],
    n_dev: Int32,
):
    var n = Int(n_dev)
    var i = Int(global_idx.x)
    if i >= n:
        return
    var raw = rebind[SIMD[type_of(z).dtype, 1]](x.reshape(row_major(n))[i] + 1)
    z.store[1](Coord(i), raw)


@extensibility.register("trait_fused_transform_bias")
struct TraitFusedTransformBias:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        # The output declares only `Output & Fused & _LegacyComputeOutput`.
        # The body branches on the form the graph compiler delivered:
        # `_LegacyComputable & TileTensorable` (the transform plus the output
        # projection for the parent store) or `Storable`.
        z: Some[Output & Fused & _LegacyComputeOutput],
        x: Some[Input & TileTensorable],
        ctx: DeviceContext,
    ) raises:
        var n = Int(z.shape().product())
        comptime if is_cpu[target]():
            comptime assert conforms_to(type_of(x), TileTensorable)
            var xt = x.to_tile_tensor[type_of(z).dtype]()
            var xf = xt.reshape(row_major(n))
            comptime if conforms_to(type_of(z), _LegacyComputable):
                comptime assert conforms_to(type_of(z), TileTensorable)
                var zt = z.to_tile_tensor()
                var zf = zt.reshape(row_major(n))
                for i in range(n):
                    var raw = xf.load[width=1]((i,)) + 1
                    var t = z.compute[1](Coord(i), raw)
                    zf.store((i,), rebind[SIMD[type_of(zt).dtype, 1]](t))
            else:
                comptime assert conforms_to(type_of(z), Storable)
                for i in range(n):
                    var raw = xf.load[width=1]((i,)) + 1
                    z.store[1](Coord(i), raw)
        else:
            comptime assert conforms_to(type_of(z), DevicePassable)
            comptime assert conforms_to(type_of(x), DevicePassable)
            comptime block_size = 64
            comptime if conforms_to(type_of(z), _LegacyComputable):
                ctx.enqueue_function[
                    _gpu_fused_transform_bias_kernel[
                        type_of(z), type_of(z).dtype, type_of(x).LayoutType
                    ]
                ](
                    z,
                    x,
                    Int32(n),
                    grid_dim=(ceildiv(n, block_size),),
                    block_dim=(block_size,),
                )
            else:
                ctx.enqueue_function[
                    _gpu_fused_transform_bias_store_form_kernel[
                        type_of(z), type_of(z).dtype, type_of(x).LayoutType
                    ]
                ](
                    z,
                    x,
                    Int32(n),
                    grid_dim=(ceildiv(n, block_size),),
                    block_dim=(block_size,),
                )


# UNIFIED-closure fused-transform variant: the device kernel binds the PROJECTED
# transform proxy (`downcast[ZHost.device_type, _LegacyComputable & TileTensorable]`,
# access-enabled) and applies the transform + final store through a unified
# closure that captures that projected value. Same context gate as store/load: the
# closure captures `device_type`, so the `compute`/`to_tile_tensor` bodies'
# `comptime assert _access_enabled` passes; a host-side transform is a compile
# error (see `trait_fused_transform_host_negative`).
def _gpu_fused_transform_bias_unified_closure_kernel[
    ZHost: DevicePassable,
    dtype: DType,
    LayoutType: TensorLayout,
](
    z: downcast[ZHost.device_type, _LegacyComputable & TileTensorable],
    x: TileTensor[dtype, LayoutType, ImmutAnyOrigin],
    n_dev: Int32,
):
    var n = Int(n_dev)
    var i = Int(global_idx.x)
    if i >= n:
        return
    var coord = Coord(i)
    var raw = rebind[SIMD[type_of(z).dtype, 1]](x.reshape(row_major(n))[i] + 1)

    @inline(.always)
    def transform_and_store(c: Coord, r: SIMD[type_of(z).dtype, 1]) {imm z}:
        var t = z.compute[1](c, r)
        var zt = z.to_tile_tensor[mut=True]()
        zt.store(c, rebind[SIMD[type_of(zt).dtype, 1]](t))

    transform_and_store(coord, raw)


@extensibility.register("trait_fused_transform_bias_unified_closure")
struct TraitFusedTransformBiasUnifiedClosure:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        z: Some[Output & Fused & _LegacyComputeOutput],
        x: Some[Input & TileTensorable],
        ctx: DeviceContext,
    ) raises:
        var n = Int(z.shape().product())
        comptime if is_cpu[target]():
            comptime assert conforms_to(type_of(x), TileTensorable)
            var xt = x.to_tile_tensor[type_of(z).dtype]()
            var xf = xt.reshape(row_major(n))
            comptime if conforms_to(type_of(z), _LegacyComputable):
                comptime assert conforms_to(type_of(z), TileTensorable)
                var zt = z.to_tile_tensor()
                var zf = zt.reshape(row_major(n))
                for i in range(n):
                    var raw = xf.load[width=1]((i,)) + 1
                    var t = z.compute[1](Coord(i), raw)
                    zf.store((i,), rebind[SIMD[type_of(zt).dtype, 1]](t))
            else:
                comptime assert conforms_to(type_of(z), Storable)
                for i in range(n):
                    var raw = xf.load[width=1]((i,)) + 1
                    z.store[1](Coord(i), raw)
        else:
            comptime assert conforms_to(type_of(z), DevicePassable)
            comptime assert conforms_to(type_of(x), DevicePassable)
            comptime block_size = 64
            comptime if conforms_to(type_of(z), _LegacyComputable):
                ctx.enqueue_function[
                    _gpu_fused_transform_bias_unified_closure_kernel[
                        type_of(z), type_of(z).dtype, type_of(x).LayoutType
                    ]
                ](
                    z,
                    x,
                    Int32(n),
                    grid_dim=(ceildiv(n, block_size),),
                    block_dim=(block_size,),
                )
            else:
                ctx.enqueue_function[
                    _gpu_fused_transform_bias_store_form_kernel[
                        type_of(z), type_of(z).dtype, type_of(x).LayoutType
                    ]
                ](
                    z,
                    x,
                    Int32(n),
                    grid_dim=(ceildiv(n, block_size),),
                    block_dim=(block_size,),
                )


# A fused-output `TensorTuple` declared before a `_LegacyComputeOutput` slot,
# so each argument's position differs from its result position. Copies `x`
# into every tuple element through its fused store, and computes `z` as in
# `trait_fused_transform_bias`.
@extensibility.register("trait_tuple_then_compute_output")
struct TraitTupleThenComputeOutput:
    @staticmethod
    def execute[
        Outs: TypeList[Trait=Output & Fused & StoreAccess, ...],
        //,
        target: StaticString,
    ](
        outputs: TensorTuple[*Outs],
        z: Some[Output & Fused & _LegacyComputeOutput],
        x: Some[Input & TileTensorable],
        ctx: DeviceContext,
    ) raises:
        comptime if is_cpu[target]():
            comptime assert conforms_to(type_of(x), TileTensorable)
            var n = Int(x.shape().product())
            var xt = x.to_tile_tensor[type_of(z).dtype]()
            var xf = xt.reshape(row_major(n))
            comptime for k in range(len(Outs)):
                comptime assert conforms_to(type_of(outputs[k]), Storable)
                for j in range(n):
                    var v = xf.load[width=1]((j,))
                    outputs[k].store[1](
                        (j,), v.cast[type_of(outputs[k]).dtype]()
                    )
            comptime if conforms_to(type_of(z), _LegacyComputable):
                comptime assert conforms_to(type_of(z), TileTensorable)
                var zt = z.to_tile_tensor()
                var zf = zt.reshape(row_major(n))
                for i in range(n):
                    var raw = xf.load[width=1]((i,)) + 1
                    var t = z.compute[1](Coord(i), raw)
                    zf.store((i,), rebind[SIMD[type_of(zt).dtype, 1]](t))
            else:
                comptime assert conforms_to(type_of(z), Storable)
                for i in range(n):
                    var raw = xf.load[width=1]((i,)) + 1
                    z.store[1](Coord(i), raw)
        else:
            raise Error("trait_tuple_then_compute_output is CPU-only")


# NEGATIVE control: transforms DIRECTLY on the fused-transform proxy at the host
# entry. On a GPU device the emitted proxy is the device-passable HOST proxy
# (`_access_enabled=False`), so the compute body's `comptime assert
# _access_enabled` fails at instantiation -- must NOT compile. Exercised
# (expecting failure) by `mogg-kernel-api-e2e-trait-fused-transform-host-negative.mlir`.
@extensibility.register("trait_fused_transform_host_negative")
struct TraitFusedTransformHostNegative:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        z: Some[Output & Fused & _LegacyComputeOutput],
        x: Some[Input & TileTensorable],
        ctx: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(z), DevicePassable)
        comptime assert conforms_to(type_of(z), _LegacyComputable)
        # Direct host-side transform on the device-passable proxy: the context
        # gate (`comptime assert _access_enabled` in the compute body) rejects it.
        _ = z.compute[1](Coord(0), SIMD[type_of(z).dtype, 1](0))


# Kernel with both a struct-level (`flip`) and fn-level (`dtype`) compile-time
# parameter, where an argument type references the fn-level param. Exercises
# two MOToMOGG paths that broke in different ways during the
# MojoLibraryAnalysis migration:
#   1. `validateCustomOpParameters` rejected legitimate calls passing a
#      struct-level param because the validation only walked fn-level params.
#   2. EmitMojo's `inferByInputs` collection mapped depth-0 `ParamIndexRefAttr`s
#      into `fnLevelParams` alone, but Mojo lays out the generator's param
#      list as `structLevelParams + fnLevelParams`, so the lookup was off by
#      `structLevelParams.size()` -- e.g. for a kernel with struct param
#      `flip` and fn params `[dtype, target]`, an arg referencing `dtype` is
#      `*(0, 1)`, which mapped to `target` (`fnLevelParams[1]`) instead of
#      `dtype` (`(struct + fn)[1]`).
@extensibility.register("imposter_add_struct_param")
struct FooWithStructParam[flip: Bool]:
    @staticmethod
    def execute[
        dtype: DType,
        //,
        target: StaticString,
    ](
        z: OutputTensor[dtype=dtype, ...],
        x: InputTensor[dtype=dtype, ...],
        ctx: DeviceContext,
    ) raises:
        @inline(.always)
        def func[width: Int](idx: Coord) {var} -> SIMD[z.dtype, width]:
            var v = rebind[SIMD[z.dtype, width]](x.load[width](idx))
            comptime if Self.flip:
                return -v
            return v

        foreach(func, z, ctx)


@extensibility.register_shape_function("imposter_add_struct_param")
def imposter_add_struct_param_shape(
    x: Some[Tensor],
) -> IndexList[type_of(x).rank]:
    return rebind[IndexList[type_of(x).rank]](coord_to_index_list(x.shape()))


def _matmul(
    c: ManagedTensorSlice[mut=True, ...],
    a: ManagedTensorSlice,
    b: ManagedTensorSlice,
) raises:
    def store_c[
        dtype: DType, width: SIMDLength, *, alignment: Int = 1
    ](idx: IndexList[2], val: SIMD[dtype, width]) {var c}:
        c.store[1](idx, rebind[Scalar[c.dtype]](val))

    _matmul[c.dtype](a, b, store_c)


def _matmul[
    StoreFnType: ImplicitlyCopyable
    & RegisterPassable
    & def[dtype: DType, width: SIMDLength, *, alignment: Int = 1](
        IndexList[2], SIMD[dtype, width]
    ) -> None,
    //,
    out_dtype: DType,
](a: ManagedTensorSlice, b: ManagedTensorSlice, store_fn: StoreFnType) raises:
    var m = a.dim_size(0)
    var n = a.dim_size(1)
    var k = b.dim_size(1)

    for i in range(m):
        for j in range(n):
            var c_val = Scalar[out_dtype](0)
            for l in range(k):
                var a_val = a.load[1]((i, l)).cast[out_dtype]()
                var b_val = b.load[1]((l, j)).cast[out_dtype]()
                c_val += a_val * b_val

            store_fn((i, j), c_val)


# c = a @ b, should support CPU and GPU
@extensibility.register("imposter_matmul")
struct ImposterMatmul:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        c: OutputTensor,
        a: InputTensor,
        b: InputTensor,
        ctx: DeviceContext,
    ) raises:
        _matmul(c, a, b)


@extensibility.register_shape_function("imposter_matmul")
def imposter_matmul_shape(
    a: Some[Tensor],
    b: Some[Tensor],
) -> IndexList[2]:
    var shape = coord_to_index_list(a.shape())
    shape[1] = coord_to_index_list(b.shape())[1]
    return rebind[IndexList[2]](shape)


@extensibility.register("print_tensor_spec")
struct PrintTensorSpecOp:
    @staticmethod
    def execute[
        target: StaticString,
    ](outp: OutputTensor, x: InputTensor, ctx: DeviceContext) raises:
        print("x.shape =", x._static_shape_tuple)
        print("x.strides =", x._static_strides_tuple)
        print("x.alignment =", x.alignment)
        print("x.address_space =", x.address_space)

        @inline(.always)
        def func[width: Int](idx: Coord) {var} -> SIMD[outp.dtype, width]:
            return rebind[SIMD[outp.dtype, width]](x.load[width](idx))

        foreach(func, outp, ctx)


@extensibility.register_shape_function("print_tensor_spec")
def print_tensor_spec_shape(x: Some[Tensor]) -> IndexList[type_of(x).rank]:
    return rebind[IndexList[type_of(x).rank]](coord_to_index_list(x.shape()))


@extensibility.register("print_tensor")
struct PrintTensor:
    @staticmethod
    def execute[
        target: StaticString,
    ](outp: OutputTensor, input: InputTensor, ctx: DeviceContext) raises:
        print(input)

        @inline(.always)
        def func[width: Int](idx: Coord) {var} -> SIMD[outp.dtype, width]:
            return rebind[SIMD[outp.dtype, width]](input.load[width](idx))

        foreach(func, outp, ctx)


@extensibility.register("print_tensor_spec_fused")
struct PrintTensorSpecFusedOp:
    @staticmethod
    def execute[
        target: StaticString,
    ](outp: FusedOutputTensor, x: FusedInputTensor, ctx: DeviceContext,) raises:
        print("x.shape =", x._static_shape_tuple)
        print("x.strides =", x._static_strides_tuple)
        print("x.alignment =", x.alignment)
        print("x.address_space =", x.address_space)

        @inline(.always)
        def func[width: Int](idx: Coord) {var} -> SIMD[outp.dtype, width]:
            return rebind[SIMD[outp.dtype, width]](x._fused_load[width](idx))

        foreach(func, outp, ctx)


@extensibility.register_shape_function("print_tensor_spec_fused")
def print_tensor_spec_fused_shape(
    x: Some[Tensor],
) -> IndexList[type_of(x).rank]:
    return rebind[IndexList[type_of(x).rank]](coord_to_index_list(x.shape()))


@extensibility.register("print_output_shape_strides_fused")
struct PrintOutputShapeStridesFusedOp:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        output: FusedOutputTensor,
        x: FusedInputTensor,
        ctx: DeviceContext,
    ) raises:
        print("output._static_shape_tuple = ", output._static_shape_tuple)
        print("output._static_strides_tuple = ", output._static_strides_tuple)
        print("output._runtime_shape = ", output.shape())
        print("output._runtime_strides = ", output.strides())

        @inline(.always)
        def func[width: Int](idx: Coord) {var} -> SIMD[output.dtype, width]:
            return (
                rebind[SIMD[output.dtype, width]](x._fused_load[width](idx))
                * 2.0
            )

        foreach(func, output, ctx)


@extensibility.register_shape_function("print_output_shape_strides_fused")
def print_output_shape_strides_fused_shape(
    x: Some[Tensor],
) -> IndexList[type_of(x).rank]:
    return rebind[IndexList[type_of(x).rank]](coord_to_index_list(x.shape()))


@extensibility.register("test_monoid_mixed_fields")
struct TestMonoidMixedFields[dtype: DType, W: Int = 1](ReduceOp):
    """Mixes output fields (`W`-wide `SIMD`) with internal state."""

    comptime Single = TestMonoidMixedFields[Self.dtype, 1]
    comptime width = Self.W

    var acc: SIMD[Self.dtype, Self.W]
    var index: SIMD[.int64, Self.W]
    var best: Scalar[Self.dtype]
    var pair: SIMD[Self.dtype, 2]
    var count: Int

    def __init__(out self):
        self.acc = 0
        self.index = 0
        self.best = 0
        self.pair = 0
        self.count = 0

    def accumulate[
        val_dtype: DType, w: Int
    ](
        mut self,
        val: SIMD[val_dtype, w],
        idx: SIMD[.int64, w] = SIMD[.int64, w](0),
    ):
        pass

    def join(mut self, other: Self):
        self.acc += other.acc

    def __getitem__(self, j: Int) -> Self.Single:
        return Self.Single()

    def __setitem__(mut self, j: Int, s: Self.Single):
        pass


@extensibility.register("test_monoid_renamed_params")
struct TestMonoidRenamedParams[T: DType, V: Int](ReduceOp):
    """Parameters not named `dtype`/`W`."""

    comptime Single = TestMonoidRenamedParams[Self.T, 1]
    comptime width = Self.V

    var total: SIMD[Self.T, Self.V]

    def __init__(out self):
        self.total = 0

    def accumulate[
        val_dtype: DType, w: Int
    ](
        mut self,
        val: SIMD[val_dtype, w],
        idx: SIMD[.int64, w] = SIMD[.int64, w](0),
    ):
        pass

    def join(mut self, other: Self):
        self.total += other.total

    def __getitem__(self, j: Int) -> Self.Single:
        return Self.Single()

    def __setitem__(mut self, j: Int, s: Self.Single):
        pass


@extensibility.register("test_monoid_defaulted_param")
struct TestMonoidDefaultedParam[dtype: DType, W: Int, K: Int = 3](ReduceOp):
    """An extra parameter with a default."""

    comptime Single = TestMonoidDefaultedParam[Self.dtype, 1, Self.K]
    comptime width = Self.W

    var acc: SIMD[Self.dtype, Self.W]

    def __init__(out self):
        self.acc = 0

    def accumulate[
        val_dtype: DType, w: Int
    ](
        mut self,
        val: SIMD[val_dtype, w],
        idx: SIMD[.int64, w] = SIMD[.int64, w](0),
    ):
        pass

    def join(mut self, other: Self):
        self.acc += other.acc

    def __getitem__(self, j: Int) -> Self.Single:
        return Self.Single()

    def __setitem__(mut self, j: Int, s: Self.Single):
        pass


@extensibility.register("imposter_add_elementwise")
struct AddElementwise:
    @staticmethod
    def elementwise[
        dtype: DType,
        width: SIMDLength,
    ](lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]) -> SIMD[dtype, width]:
        return lhs + rhs


@extensibility.register("imposter_add_lhs")
struct AddFuseLHS:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        z: OutputTensor,
        x: FusedInputTensor,
        y: InputTensor,
        ctx: DeviceContext,
    ) raises:
        @inline(.always)
        def func[width: Int](idx: Coord) {var} -> SIMD[z.dtype, width]:
            return rebind[SIMD[z.dtype, width]](
                x._fused_load[width](idx)
            ) + rebind[SIMD[z.dtype, width]](y.load[width](idx))

        foreach(func, z, ctx)


@extensibility.register("imposter_add_fuse_inputs")
struct AddFuseInputs:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        z: OutputTensor,
        x: FusedInputTensor,
        y: FusedInputTensor,
        ctx: DeviceContext,
    ) raises:
        @inline(.always)
        def func[width: Int](idx: Coord) {var} -> SIMD[z.dtype, width]:
            return rebind[SIMD[z.dtype, width]](
                x._fused_load[width](idx)
            ) + rebind[SIMD[z.dtype, width]](y._fused_load[width](idx))

        foreach(func, z, ctx)


@extensibility.register("imposter_add_compute_fusion")
struct AddComputeFusion:
    @staticmethod
    def execute[
        dtype: DType,
        target: StaticString,
    ](
        z: _FusedComputeOutputTensor[dtype=dtype, rank=2, ...],
        x: InputTensor[dtype=dtype, rank=2, ...],
        y: InputTensor[dtype=dtype, rank=2, ...],
        ctx: DeviceContext,
    ) raises:
        comptime has_compute_lambda = type_of(z)._has_compute_fusion
        print("has_compute_lambda =", has_compute_lambda)

        @inline(.always)
        def func[
            width: Int, alignment: Int = 1
        ](idx: Coord) {var x, var y, var z}:
            var val = rebind[SIMD[dtype, width]](x.load[width](idx)) + rebind[
                SIMD[dtype, width]
            ](y.load[width](idx))
            comptime if has_compute_lambda:
                val = z._fused_compute_output_lambda[
                    width=width, element_alignment=alignment
                ](idx, val)
                z.store[element_alignment=alignment](idx, val)
            else:
                z._fused_store[element_alignment=alignment](idx, val)

        elementwise[1, target=target](func, z.shape_coord(), ctx)


@extensibility.register_shape_function("imposter_add_compute_fusion")
def imposter_add_compute_fusion_shape(
    x: Some[Tensor], y: Some[Tensor]
) -> IndexList[type_of(x).rank]:
    return rebind[IndexList[type_of(x).rank]](coord_to_index_list(x.shape()))


@extensibility.register("imposter_add_fuse_output")
struct AddFuseOutput:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        z: FusedOutputTensor,
        x: InputTensor,
        y: InputTensor,
        ctx: DeviceContext,
    ) raises:
        @inline(.always)
        def func[width: Int](idx: Coord) {var} -> SIMD[z.dtype, width]:
            return rebind[SIMD[z.dtype, width]](x.load[width](idx)) + rebind[
                SIMD[z.dtype, width]
            ](y.load[width](idx))

        foreach[target=target](func, z, ctx)


# c = a @ b, should support CPU and GPU
@extensibility.register("matmul_fuse_out")
struct MatmulFuseOut:
    @staticmethod
    def execute[
        target: StaticString,
        has_epilogue_fusion: Bool,
    ](
        c: FusedOutputTensor,
        a: InputTensor,
        b: InputTensor,
        ctx: DeviceContext,
    ) raises:
        comptime rank = a.rank
        comptime a_dtype = a.dtype
        comptime b_dtype = b.dtype
        comptime c_dtype = c.dtype

        @inline(.always)
        def out_func[
            dtype: DType, width: SIMDLength, *, alignment: Int = 1
        ](idx: IndexList[2], val: SIMD[dtype, width]) {var c}:
            c._fused_store(idx, rebind[SIMD[c.dtype, width]](val))

        print("has_epilogue_fusion =", has_epilogue_fusion)

        _matmul[c.dtype](a, b, out_func)


@extensibility.register_shape_function("matmul_fuse_out")
def matmul_fuse_out_shape(
    a: Some[Tensor],
    b: Some[Tensor],
) -> IndexList[2]:
    var shape = coord_to_index_list(a.shape())
    shape[1] = coord_to_index_list(b.shape())[1]
    return rebind[IndexList[2]](shape)


# Simple, expects variadics to have the same size, and simply copies the first
# number from the associated inputs to outputs, plus a bias
@extensibility.register("variadic_input_to_output")
struct VariadicInputToOutput:
    @staticmethod
    def execute[
        dtype: DType,
        size: Int,
        target: StaticString,
    ](
        output: OutputVariadicTensors[dtype=dtype, rank=1, size=size, ...],
        bias: InputTensor[dtype=dtype, rank=1, ...],
        input: InputVariadicTensors[dtype=dtype, rank=1, size=size, ...],
    ):
        comptime for i in range(size):
            for j in range(input[i].size()):
                output[i][j] = input[i][j]
            output[i][0] += bias[0]


# Simply adds the first number of bias to the first number of boath outputs
# Mainly here to test logic with multiple DPS outputs
@extensibility.register("add_bias_to_two_tensors")
struct AddBiasToDouble:
    @staticmethod
    def execute[
        rank: Int,
        dtype: DType,
    ](
        output1: OutputTensor[dtype=dtype, rank=rank, ...],
        output2: OutputTensor[dtype=dtype, rank=rank, ...],
        input1: InputTensor[dtype=dtype, rank=rank, ...],
        input2: InputTensor[dtype=dtype, rank=rank, ...],
        bias: InputTensor[dtype=dtype, rank=rank, ...],
    ):
        output1[0] = input1[0] + bias[0]
        output2[0] = input2[0] + bias[0]


@extensibility.register("inplace_increment_elem")
struct BasicInplace:
    @staticmethod
    def execute[
        dtype: DType,
    ](input: MutableInputTensor[dtype=dtype, rank=2, ...]):
        var x = input[0, 0]
        x += 1
        input[0, 0] = x


# Have this nearly identical version as having a raise changes the Mojo function's signature
@extensibility.register("inplace_increment_elem_raises")
struct BasicInplaceRaises:
    @staticmethod
    def execute[
        dtype: DType,
    ](input: MutableInputTensor[dtype=dtype, rank=2, ...]) raises:
        var x = input[0, 0]
        x += 1
        input[0, 0] = x


@extensibility.register("variadic_add")
struct VariadicAdd:
    @staticmethod
    def execute[
        dtype: DType,
        rank: Int,
        target: StaticString,
    ](
        output: OutputTensor[dtype=dtype, rank=rank, ...],
        inputs: FusedInputVariadicTensors[dtype=dtype, rank=rank, ...],
        ctx: DeviceContext,
    ) raises:
        @inline(.always)
        def func[width: Int](idx: Coord) {var} -> SIMD[dtype, width]:
            var acc = SIMD[dtype, width](0)

            comptime for i in range(inputs.size):
                acc += inputs[i]._fused_load[width](idx)

            return acc

        foreach(func, output, ctx)


# Shaped like `mo.concat`, which the graph compiler fuses as a view: an opaque
# kernel taking fused inputs and a fused output, for lit tests of prologue and
# epilogue fusion into one. They never run it, so it computes nothing.
@extensibility.register("fused_variadic_inputs")
struct FusedVariadicInputs:
    @staticmethod
    def execute[
        dtype: DType,
        rank: Int,
        target: StaticString,
        axis: Int,
    ](
        output: FusedOutputTensor[dtype=dtype, rank=rank, ...],
        inputs: FusedInputVariadicTensors[dtype=dtype, rank=rank, ...],
        ctx: DeviceContext,
    ) raises:
        pass


@extensibility.register("print_shape_fused")
struct ElementwisePrintShape:
    @staticmethod
    def execute[
        target: StaticString,
    ](z: FusedOutputTensor, x: FusedInputTensor, ctx: DeviceContext,) raises:
        @inline(.always)
        def func[width: Int](idx: Coord) {var} -> SIMD[z.dtype, width]:
            return rebind[SIMD[z.dtype, width]](x._fused_load[width](idx))

        print("input.shape =", x.shape())
        print("output.shape =", z.shape())

        foreach(func, z, ctx)


@extensibility.register_shape_function("print_shape_fused")
def print_shape_fused_shape(x: Some[Tensor]) -> IndexList[type_of(x).rank]:
    return rebind[IndexList[type_of(x).rank]](coord_to_index_list(x.shape()))


# Raises if input shape is 10
@extensibility.register("custom_op_that_raises")
struct CustomOpThatRaises:
    @staticmethod
    def execute[
        target: StaticString,
    ](z: OutputTensor, x: InputTensor, ctx: DeviceContext,) raises:
        if x.shape()[0] == 10:
            raise ("input_shape[0] == 10")

        @inline(.always)
        def func[width: Int](idx: Coord) {var} -> SIMD[z.dtype, width]:
            return rebind[SIMD[z.dtype, width]](x._fused_load[width](idx))

        foreach(func, z, ctx)


@extensibility.register_shape_function("custom_op_that_raises")
def custom_op_that_raises_shape(
    x: Some[Tensor],
) raises -> IndexList[type_of(x).rank]:
    print("Hello")
    var out_shape = coord_to_index_list(x.shape())
    if out_shape[0] == 20:
        raise ("data.get_shape()[0] == 20")
    return rebind[IndexList[type_of(x).rank]](out_shape)


@extensibility.register("mo.test.failing_constraint")
struct OpThatAlwaysFailsConstraint:
    @staticmethod
    def execute[
        dtype: DType, rank: Int
    ](
        out_tensor: OutputTensor[dtype=dtype, rank=rank, ...],
        in_tensor: InputTensor[dtype=dtype, rank=rank, ...],
    ):
        comptime assert (
            1 == 2
        ), "Expected constraint failure for error message testing"


@extensibility.register("mo.test.return_error")
struct OpThatAlwaysRaises:
    @staticmethod
    def execute[
        dtype: DType, rank: Int
    ](
        out_tensor: OutputTensor[dtype=dtype, rank=rank, ...],
        in_tensor: InputTensor[dtype=dtype, rank=rank, ...],
    ) raises:
        out_tensor[0] = in_tensor[0]
        raise Error("This is an error")


@extensibility.register("monnx.abs_v13")
struct MONNXAbsOverload:
    @staticmethod
    def execute[
        dtype: DType, rank: Int
    ](
        out_tensor: OutputTensor[dtype=dtype, rank=rank, ...],
        in_tensor: InputTensor[dtype=dtype, rank=rank, ...],
        ctx: DeviceContext,
    ) raises:
        @inline(.always)
        def func[width: Int](idx: Coord) {var} -> SIMD[dtype, width]:
            return abs(in_tensor._fused_load[width](idx))

        print("The custom identity op is running!")
        foreach(func, out_tensor, ctx)


@extensibility.register("torch.aten.abs")
struct MTorchAbsOverload:
    @staticmethod
    def execute[
        dtype: DType, rank: Int
    ](
        out_tensor: OutputTensor[dtype=dtype, rank=rank, ...],
        in_tensor: InputTensor[dtype=dtype, rank=rank, ...],
        ctx: DeviceContext,
    ) raises:
        @inline(.always)
        def func[width: Int](idx: Coord) {var} -> SIMD[dtype, width]:
            return abs(in_tensor._fused_load[width](idx))

        print("The custom identity op is running!")
        foreach(func, out_tensor, ctx)


@extensibility.register("op_with_custom_params")
struct OpWithCustomParams:
    @staticmethod
    def execute[
        dtype: DType,
        rank: Int,
        custom_int: Int,
        custom_i32: Int32,
        custom_str: StaticString,
        custom_dtype: DType,
        custom_float: Float32,
    ](
        out_tensor: OutputTensor[dtype=dtype, rank=rank, ...],
        in_tensor: InputTensor[dtype=dtype, rank=rank, ...],
    ) raises:
        out_tensor[0] = in_tensor[0]
        print("custom_int =", custom_int)
        print("custom_i32 =", custom_i32)
        print("custom_str =", custom_str)
        print("custom_dtype =", custom_dtype)
        print("custom_float =", custom_float)


@extensibility.register("mgprt_test_func")
struct MGPRTTestFunc:
    @staticmethod
    def execute(out_tensor: OutputTensor) raises:
        external_call["MGP_RT_TEST", NoneType]()


@extensibility.register("mutable_test_op")
struct MutableTestOp:
    @staticmethod
    def execute(in_place_tensor: MutableInputTensor) raises:
        in_place_tensor._ptr.unsafe_store(0, 0)


# For testing support for Scalar[...] in Mojo
@extensibility.register("supports_scalar_kernel")
struct SupportsScalarKernel:
    @staticmethod
    def execute[
        dtype: DType
    ](
        outp: OutputTensor[dtype=dtype, rank=1, ...],
        x: InputTensor[dtype=dtype, rank=1, ...],
        y: Scalar[dtype],
    ) raises:
        print("datatype is", dtype)
        print("value is", y)


@extensibility.register("trait_supports_scalar_kernel")
struct TraitSupportsScalarKernel:
    @staticmethod
    def execute[
        dtype: DType,
        Z: Output & TileTensorable,
        X: Input & TileTensorable,
    ](z: Z, x: X, y: Scalar[dtype],) raises:
        print("datatype is", dtype)
        print("value is", y)


@extensibility.register("kernel_with_no_target")
struct KernelWithNoTarget:
    @staticmethod
    def execute[
        dtype: DType
    ](
        outp: OutputTensor[dtype=dtype, ...],
        x: InputTensor[dtype=dtype, ...],
    ) raises:
        print("hello from kernel with no target")


@extensibility.register("basic_target")
struct BasicTarget:
    @staticmethod
    def execute[
        dtype: DType, target: StaticString
    ](
        outp: OutputTensor[dtype=dtype, ...],
        x: InputTensor[dtype=dtype, ...],
    ) raises:
        print("hello from kernel on", target)


struct MyCustomScalarReg[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    var val: Scalar[Self.dtype]

    @implicit
    def __init__(out self, val: Scalar[Self.dtype]):
        print("MyCustomScalarReg.__init__", val)
        self.val = val

    def __deinit__(deinit self):
        print("MyCustomScalarReg.__deinit__", self.val)


@extensibility.register("buff_to_my_custom_scalar_reg")
struct BuffToMyCustomScalarReg:
    @staticmethod
    def execute[
        target: StaticString
    ](x: InputTensor[dtype=.int32, rank=1, ...]) -> MyCustomScalarReg[
        DType.int32
    ]:
        return MyCustomScalarReg(x[0])


@extensibility.register("my_custom_scalar_reg_to_buff")
struct CustomScalarRegToBuff:
    @staticmethod
    def execute[
        target: StaticString
    ](
        input: OutputTensor[dtype=.int32, rank=1, ...],
        x: MyCustomScalarReg[.int32],
    ):
        input[0] = x.val


@extensibility.register("test_custom_op")
struct TestCustomOp:
    @staticmethod
    def execute[
        target: StaticString, dtype: DType, rank: Int
    ](
        output: OutputTensor[dtype=dtype, rank=rank, ...],
        input: InputTensor[dtype=dtype, rank=rank, ...],
    ):
        print("World!")


@extensibility.register_shape_function("test_custom_op")
def test_custom_op_shape(input: Some[Tensor]) -> IndexList[type_of(input).rank]:
    print("Hello")
    return rebind[IndexList[type_of(input).rank]](
        coord_to_index_list(input.shape())
    )


# A custom op whose shape function is trait-typed: a metadata-only, no-role
# `Some[TileTensorable]` argument that reads only `.shape()`. This exercises the
# trait shape-fn emit path (the operand is constructed as a `_MoggShapeTensorArg`
# instead of a legacy `InputTensor`). The `execute` entry stays on the legacy
# lane -- the shape-fn migration is independent of how `execute` is written --
# and copies input to output so the result is verifiable end-to-end.
@extensibility.register("trait_metadata_shape_op")
struct TraitMetadataShapeOp:
    @staticmethod
    def execute[
        target: StaticString, dtype: DType, rank: Int
    ](
        output: OutputTensor[dtype=dtype, rank=rank, ...],
        input: InputTensor[dtype=dtype, rank=rank, ...],
        ctx: DeviceContext,
    ) raises:
        @inline(.always)
        def func[width: Int](idx: Coord) {var} -> SIMD[output.dtype, width]:
            return input.load[width](idx)

        foreach(func, output, ctx)


@extensibility.register_shape_function("trait_metadata_shape_op")
def trait_metadata_shape_op_shape[
    T: TileTensorable
](input: T) -> IndexList[T.rank]:
    return rebind[IndexList[T.rank]](coord_to_index_list(input.shape()))


# A custom op whose shape function binds the bare `Tensor` metadata trait
# (shape only, no layout/alignment), unlike `trait_metadata_shape_op` above
# which binds `TileTensorable`. This exercises the `_data_visible=False` emit
# path -- the emitted `_MoggShapeTensorArg` conforms only to `Tensor`, so it
# cannot see layout or read elements. `execute` stays on the legacy lane and
# copies input to output so the result is verifiable end-to-end.
@extensibility.register("trait_tensor_shape_op")
struct TraitTensorShapeOp:
    @staticmethod
    def execute[
        target: StaticString, dtype: DType, rank: Int
    ](
        output: OutputTensor[dtype=dtype, rank=rank, ...],
        input: InputTensor[dtype=dtype, rank=rank, ...],
        ctx: DeviceContext,
    ) raises:
        @inline(.always)
        def func[width: Int](idx: Coord) {var} -> SIMD[output.dtype, width]:
            return input.load[width](idx)

        foreach(func, output, ctx)


@extensibility.register_shape_function("trait_tensor_shape_op")
def trait_tensor_shape_op_shape[T: Tensor](input: T) -> IndexList[T.rank]:
    return rebind[IndexList[T.rank]](coord_to_index_list(input.shape()))


# A custom op whose shape function reads element values off a
# `TileTensorable` argument (the `sizes` input) via `to_tile_tensor()`, rather
# than just its metadata -- shape functions may read data since they always
# execute on the host. The `data` argument binds the bare `Tensor` metadata
# trait, mixing tiers within one shape function for extra coverage. `execute`
# stays on the legacy lane and copies `data` into `output` at matching
# coordinates, so the produced values are verifiable end-to-end even though
# the output shape comes from `sizes`, not from `data`'s own shape.
@extensibility.register("trait_reshape_shape_op")
struct TraitReshapeShapeOp:
    @staticmethod
    def execute[
        target: StaticString, dtype: DType
    ](
        output: OutputTensor[dtype=dtype, rank=2, ...],
        data: InputTensor[dtype=dtype, rank=2, ...],
        sizes: InputTensor[dtype=.int64, rank=1, ...],
        ctx: DeviceContext,
    ) raises:
        @inline(.always)
        def func[width: Int](idx: Coord) {var} -> SIMD[output.dtype, width]:
            return data.load[width](idx)

        foreach(func, output, ctx)


@extensibility.register_shape_function("trait_reshape_shape_op")
def trait_reshape_shape_op_shape[
    D: Tensor, S: TileTensorable
](data: D, sizes: S) -> IndexList[2]:
    var new_shape = sizes.to_tile_tensor()
    var shape = IndexList[2]()
    shape[0] = Int(new_shape[0])
    shape[1] = Int(new_shape[1])
    return shape


# A custom op whose shape function takes its variadic operands as a
# `TensorTuple` of bare-`Tensor` elements, so the tuple holds one emitted
# `_MoggShapeTensorArg` per operand by value. `execute` copies the first input
# to the output. CPU only.
@extensibility.register("trait_tensor_tuple_shape_op")
struct TraitTensorTupleShapeOp:
    @staticmethod
    def execute[
        Ins: TypeList[Trait=Input & TileTensorable, ...],
        //,
        target: StaticString,
    ](
        output: Some[Output & TileTensorable],
        inputs: TensorTuple[*Ins],
        ctx: DeviceContext,
    ) raises:
        comptime if not is_cpu[target]():
            raise Error("trait_tensor_tuple_shape_op only runs on the CPU")
        comptime assert conforms_to(type_of(output), TileTensorable)
        comptime assert conforms_to(type_of(inputs[0]), TileTensorable)
        var zt = output.to_tile_tensor()
        var xt = inputs[0].to_tile_tensor[type_of(output).dtype]()
        var n = Int(output.shape().product())
        var zf = zt.reshape(row_major(n))
        var xf = xt.reshape(row_major(n))
        for j in range(n):
            zf.store((j,), xf.load((j,)))


@extensibility.register_shape_function("trait_tensor_tuple_shape_op")
def trait_tensor_tuple_shape_op_shape[
    Ts: TypeList[Trait=Tensor, ...], //
](inputs: TensorTuple[*Ts]) -> IndexList[Ts[0].rank]:
    return rebind[IndexList[Ts[0].rank]](coord_to_index_list(inputs[0].shape()))


@extensibility.register("single_device_context")
struct SingleDeviceContext:
    @staticmethod
    def execute[
        dtype: DType
    ](
        output: OutputTensor[dtype=dtype, ...],
        x: InputTensor[dtype=dtype, ...],
        dev_ctx: DeviceContextArray[1],
    ) raises:
        dev_ctx[0].synchronize()


@extensibility.register("multi_device_context")
struct MultiDeviceContext:
    @staticmethod
    def execute[
        dtype: DType
    ](
        output: OutputTensor[dtype=dtype, ...],
        x: InputTensor[dtype=dtype, ...],
        dev_ctxs: DeviceContextArray[2],
    ) raises:
        print("dev_ctx0.id() =", dev_ctxs[0].id())
        print("dev_ctx1.id() =", dev_ctxs[1].id())
        dev_ctxs[0].synchronize()
        dev_ctxs[1].synchronize()


@extensibility.register("multi_device_context_dedup")
struct MultiDeviceContextDedup:
    @staticmethod
    def execute[
        dtype: DType
    ](
        output: OutputTensor[dtype=dtype, ...],
        x: InputTensor[dtype=dtype, ...],
        y: InputTensor[dtype=dtype, ...],
        dev_ctxs: DeviceContextArray[2],
    ) raises:
        dev_ctxs[0].synchronize()
        dev_ctxs[1].synchronize()


@extensibility.register("variadic_device_context")
struct VariadicDeviceContext:
    @staticmethod
    def execute[
        dtype: DType,
        rank: Int,
        target: StaticString = "gpu",
    ](
        outputs: OutputVariadicTensors[dtype=dtype, rank=rank, ...],
        inputs: InputVariadicTensors[dtype=dtype, rank=rank, ...],
        dev_ctxs: DeviceContextArray,
    ) raises:
        for i in range(len(dev_ctxs)):
            print("dev_ctxs[", i, "].id() =", dev_ctxs[i].id())
            dev_ctxs[i].synchronize()


@extensibility.register("imposter_cast_elementwise")
struct CastElementwise:
    @staticmethod
    def elementwise[
        dtype: DType,
        out_dtype: DType,
        width: SIMDLength,
    ](x: SIMD[dtype, width]) -> SIMD[out_dtype, width]:
        return x.cast[out_dtype]()


@extensibility.register("print_vector_size")
struct PrintVectorSize:
    @staticmethod
    def elementwise[
        dtype: DType,
        width: SIMDLength,
    ](x: SIMD[dtype, width], idx: IndexList) -> SIMD[dtype, width]:
        if idx[0] == 0:
            # Only print once
            # We can't directly print `width` as it's HW-dependent.
            # Instead compare it with the width of i8 / f32.
            print(
                "width == simd_width_of[DType.int8]():",
                width == simd_width_of[DType.int8](),
            )
            print(
                "width == simd_width_of[DType.float32]():",
                width == simd_width_of[DType.float32](),
            )
        return x


@extensibility.register("tensor_opaque_tensor_kernel")
struct TensorOpaqueTensorKernel:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        output1: OutputTensor,
        output2: OutputTensor,
        opaque: MyCustomScalarSI32,
        input: InputTensor,
    ) raises:
        pass


@extensibility.register("test_same_rank_dtype")
struct TestSameRankDtype:
    @staticmethod
    def execute[
        dtype: DType, rank: Int
    ](
        output: OutputTensor[dtype=dtype, rank=rank, ...],
        input: InputTensor[dtype=dtype, rank=rank, ...],
    ):
        output[0] = input[0]


@extensibility.register("test_fix_rank")
struct TestFixRank:
    @staticmethod
    def execute[
        dtype: DType
    ](
        output: OutputTensor[dtype=dtype, rank=2, ...],
        input: InputTensor[dtype=dtype, rank=2, ...],
    ):
        output[0] = input[0]


@extensibility.register("test_simd_input_size_1")
struct TestSIMDInputSize1:
    @staticmethod
    def execute[
        dtype: DType
    ](
        output: OutputTensor[dtype=dtype, rank=2, ...],
        input: SIMD[dtype=dtype, length=1],
    ):
        output[0] = input[0]


@extensibility.register("test_simd_input")
struct TestSIMDInput:
    @staticmethod
    def execute[
        dtype: DType
    ](
        output: OutputTensor[dtype=dtype, rank=2, ...],
        input: SIMD[dtype=dtype, ...],
    ):
        output[0] = input[0]


@extensibility.register("test_simd_input_size_2")
struct TestSIMDInputSize2:
    @staticmethod
    def execute[
        dtype: DType
    ](
        output: OutputTensor[dtype=dtype, rank=2, ...],
        input: SIMD[dtype=dtype, length=2],
    ):
        output[0] = input[0]


@extensibility.register("custom_op_with_custom_parameter_shape_func")
struct CustomOpWithCustomParameterShapeFunc:
    @staticmethod
    def execute[
        test_flag: Int
    ](y: FusedOutputTensor, x: FusedInputTensor) raises:
        raise "This should not run!"


@extensibility.register_shape_function(
    "custom_op_with_custom_parameter_shape_func"
)
def custom_op_with_custom_parameter_shape_func_shape[
    test_flag: Int
](input_tensor: Some[Tensor]) -> IndexList[type_of(input_tensor).rank]:
    return rebind[IndexList[type_of(input_tensor).rank]](
        coord_to_index_list(input_tensor.shape())
    )


@extensibility.register("custom_op_with_mixed_bound_shape_func")
struct CustomOpWithMixedBoundShapeFunc:
    @staticmethod
    def execute[
        target: StaticString
    ](
        output: OutputTensor[rank=2, ...],
        metadata_only: InputTensor[rank=2, ...],
        value_read: InputTensor[rank=1, ...],
        num_rows: Int32,
        ctx: DeviceContext,
    ) raises:
        raise "This should not run!"


@extensibility.register_shape_function("custom_op_with_mixed_bound_shape_func")
def custom_op_with_mixed_bound_shape_func_shape(
    metadata_only: Some[Tensor],
    value_read: Some[TileTensorable],
    num_rows: Int32,
) -> IndexList[2]:
    _ = value_read.to_tile_tensor()
    var cols = coord_to_index_list(metadata_only.shape())
    return IndexList[2](Int(num_rows), cols[1])


@extensibility.register("custom_op_with_static_string_params")
struct CustomOpWithStaticStringParams:
    @staticmethod
    def execute[target: StaticString, foo: StaticString]():
        print(target)
        print(foo)


comptime _WrapperLayout = TileLayout[
    shape_types=Coord[Int64, Int64].element_types,
    stride_types=Coord[Int64, Int64].element_types,
]


@fieldwise_init
struct TensorWrapper(RegisterPassable):
    var buffer: TileTensor[
        .float32, origin=MutUntrackedOrigin, LayoutType=_WrapperLayout
    ]


@extensibility.register("to_tensor_wrapper")
struct ToTensorWrapper:
    @inline(.always)
    @staticmethod
    def execute[
        spec: StaticTensorSpec[dtype=DType.float32, rank=2, ...], //
    ](tensor: InputTensor[static_spec=spec]) -> TensorWrapper:
        var shape = tensor.shape()
        var strides = tensor.strides()
        var shape_coord = Coord[Int64, Int64]()
        var stride_coord = Coord[Int64, Int64]()
        shape_coord[0] = Int64(shape[0])
        shape_coord[1] = Int64(shape[1])
        stride_coord[0] = Int64(strides[0])
        stride_coord[1] = Int64(strides[1])
        return TensorWrapper(
            TileTensor(
                tensor.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
                _WrapperLayout(shape_coord, stride_coord),
            )
        )


@extensibility.register("print_tensor_wrapper")
struct TensorWrapperPrinter:
    @inline(.always)
    @staticmethod
    def execute(
        tw: TensorWrapper,
    ):
        print(tw.buffer)


@extensibility.register("fused_write_to_tensor_wrapper")
struct MutableStore:
    @staticmethod
    def execute[
        target: StaticString,
        _trace_name: StaticString,
    ](
        tw: TensorWrapper,
        tensor: FusedInputTensor[dtype=.float32, rank=2, ...],
        ctx: DeviceContext,
    ) raises:
        var buffer = tw.buffer

        @inline(.always)
        def out_func[
            width: Int
        ](index: IndexList[tensor.rank]) {var tensor, var buffer}:
            var idx = rebind[IndexList[tensor.rank]](index)
            var val = rebind[SIMD[.float32, width]](
                tensor._fused_load[width](idx)
            )
            idx[0] += 1
            idx[1] += 1
            buffer.store[width=width](Coord(idx), val)

        foreach_out_func[target=target, _trace_name=_trace_name](
            out_func, tensor, ctx
        )


@extensibility.register("my_square_compute_out_fusion")
struct MySquareComputeOutFusion:
    @staticmethod
    def execute[
        dtype: DType
    ](
        outp: _FusedComputeOutputTensor[dtype=dtype, rank=1, ...],
        input: InputTensor[dtype=dtype, rank=1, ...],
    ):
        var res = input[0] * input[0]
        comptime has_compute_lambda = type_of(outp)._has_compute_fusion
        comptime has_store_lambda = type_of(outp)._has_output_store_fusion
        print("has_compute_lambda =", has_compute_lambda)
        print("has_store_lambda =", has_store_lambda)

        comptime if has_compute_lambda:
            outp[0] = outp._fused_compute_output_lambda[width=1]((0,), res)
        else:
            # We still need to support normal fusion in case the kernel is fused with a cast.
            outp._fused_store[width=1]((0,), res)


@extensibility.register("inplace_no_out_func")
struct InplaceNoOutFunc:
    # This kernel is meant to test that epilogue fusion will
    # be skipped if the out_func overload of foreach is not used.
    @staticmethod
    def execute(
        output: MutableInputTensor,
        input: FusedInputTensor[dtype=output.dtype, rank=output.rank, ...],
        ctx: DeviceContext,
    ) raises:
        @inline(.always)
        def func[width: Int](idx: Coord) {var} -> SIMD[output.dtype, width]:
            var val = input._fused_load[width](idx)
            return val * SIMD[output.dtype, width](2.0)

        foreach(func, output, ctx)


@fieldwise_init
struct SIMDPair[S0: Int, S1: Int](RegisterPassable):
    var x: SIMD[.int32, Self.S0]
    var y: SIMD[.int32, Self.S1]


@extensibility.register("make_simd_pair")
struct MakeSimdPair:
    @staticmethod
    def execute[P0: Int, P1: Int]() -> SIMDPair[P0, P1]:
        return SIMDPair[P0, P1](iota[.int32, P0](), iota[.int32, P1](Int32(P0)))


@extensibility.register("kernel_with_parameterized_opaque")
struct ParameterizedOpaqueType:
    @staticmethod
    def execute[
        P0: Int
    ](output: OutputTensor[dtype=.int32, rank=1, ...], x: SIMDPair[P0, _],):
        output.store((0,), x.x)
        output.store((P0,), x.y)


@extensibility.register_shape_function("kernel_with_parameterized_opaque")
def kernel_with_parameterized_opaque_shape[
    P0: Int
](x: SIMDPair[P0, _]) -> IndexList[1]:
    return IndexList[1](x.S0 + x.S1)


@extensibility.register("kernel_with_default_parameter")
struct KernelWithDefaultParameter:
    @staticmethod
    def execute[
        rank: Int,
        rank_defaulted: Int = 128,
        rank_derived_from_other: Int = rank + 1,
        rank_derived_from_derived: Int = rank_derived_from_other + 1,
    ](x: OutputTensor[dtype=.int32, rank=1, ...], y: InputTensor):
        print("rank =", rank)
        print("rank_defaulted =", rank_defaulted)
        print("rank_derived_from_other =", rank_derived_from_other)
        print("rank_derived_from_derived =", rank_derived_from_derived)
        x.store((0,), Int32(0))


@extensibility.register("unique-parameters")
struct UniqueParameters:
    @inline(.always)
    @staticmethod
    def execute[
        dtype: DType,
        num_heads: Int,
        head_dim: Int,
        page_size: Int,
        target: StaticString,
    ](
        blocks: InputTensor[dtype=dtype, rank=6, ...],
        cache_lengths: InputTensor[dtype=.uint32, rank=1, ...],
        lookup_table: InputTensor[dtype=.uint32, rank=2, ...],
        max_lengths: InputTensor[dtype=.uint32, rank=2, ...],
    ) -> MyCustomScalarRegTrivialSI32:
        comptime if page_size:
            return MyCustomScalarRegTrivialSI32(0)
        else:
            return MyCustomScalarRegTrivialSI32(1)


@extensibility.register("shape_func_with_mutable")
struct ShapeFuncWithMutable:
    @staticmethod
    def execute(
        output: OutputTensor[rank=2, ...],
        x: MutableInputTensor[dtype=output.dtype, rank=2, ...],
    ):
        x[0, 0] = 2.0
        output[0, 0] = 1.0


@extensibility.register_shape_function("shape_func_with_mutable")
def shape_func_with_mutable_shape(
    x: Some[Tensor],
) -> IndexList[2]:
    comptime assert type_of(x).rank == 2, "x must be rank 2"
    return rebind[IndexList[2]](coord_to_index_list(x.shape()))


@extensibility.register("elemwise_add_with_param")
struct ElemwiseAddWithParam:
    @staticmethod
    def elementwise[
        dtype: DType,
        width: SIMDLength,
        bias: Int,
    ](input: SIMD[dtype, width]) -> SIMD[dtype, width]:
        return input + Scalar[dtype](bias)


@extensibility.register("elemwise_add_with_index")
struct ElemwiseAddWithIndex:
    @staticmethod
    def elementwise[
        dtype: DType,
        width: SIMDLength,
        rank: Int,
    ](input: SIMD[dtype, width], indices: IndexList[rank]) -> SIMD[
        dtype, width
    ]:
        var bias = SIMD[dtype, width]()
        comptime for i in range(width):
            bias[i] = Scalar[dtype](indices[0] + i)
        return input + bias


# Kernel that exposes only an `execute` entry point. Device-graph synthesis
# must record it through the builder's recording context, since there is no
# `build_graph` method to call.
@extensibility.register("imposter_execute_only")
struct ExecuteOnly:
    @staticmethod
    def execute[
        dtype: DType, rank: Int
    ](
        output: OutputTensor[dtype=dtype, rank=rank, ...],
        x: InputTensor[dtype=dtype, rank=rank, ...],
        ctx: DeviceContext,
    ):
        pass


# Kernel that exposes both execute and build_graph entry points.
@extensibility.register("imposter_build_graph_capable")
struct BuildGraphCapable:
    @staticmethod
    def execute[
        dtype: DType, rank: Int
    ](
        output: OutputTensor[dtype=dtype, rank=rank, ...],
        x: InputTensor[dtype=dtype, rank=rank, ...],
        ctx: DeviceContext,
    ):
        pass

    @staticmethod
    def build_graph[
        dtype: DType, rank: Int
    ](
        output: OutputTensor[dtype=dtype, rank=rank, ...],
        x: InputTensor[dtype=dtype, rank=rank, ...],
        builder: DeviceGraphBuilder,
    ):
        pass


# Fused-output sibling of `imposter_build_graph_capable`: both entry points
# declare the output `FusedOutputTensor` and take `has_epilogue_fusion`, so an
# epilogue MAP fuses into the kernel records through `build_graph` the same
# way it executes. Signature-only stub.
@extensibility.register("imposter_fused_output_build_graph")
struct FusedOutputBuildGraph:
    @staticmethod
    def execute[
        dtype: DType, rank: Int, has_epilogue_fusion: Bool
    ](
        output: FusedOutputTensor[dtype=dtype, rank=rank, ...],
        x: InputTensor[dtype=dtype, rank=rank, ...],
        ctx: DeviceContext,
    ):
        pass

    @staticmethod
    def build_graph[
        dtype: DType, rank: Int, has_epilogue_fusion: Bool
    ](
        output: FusedOutputTensor[dtype=dtype, rank=rank, ...],
        x: InputTensor[dtype=dtype, rank=rank, ...],
        builder: DeviceGraphBuilder,
    ):
        pass


# Kernel whose `build_graph` reads its variadic peers through stable address
# slots (`StableInputVariadicTensors`) while `execute` takes the plain
# pointer-backed pack. Mirrors the shape of the bundled collectives; the bodies
# are no-op stubs for import / verify / emit tests.
@extensibility.register("imposter_stable_build_graph")
struct StableBuildGraph:
    @staticmethod
    def execute[
        dtype: DType, rank: Int
    ](
        output: OutputTensor[dtype=dtype, rank=rank, ...],
        inputs: InputVariadicTensors[dtype=dtype, rank=rank, ...],
        ctx: DeviceContext,
    ):
        pass

    @staticmethod
    def build_graph[
        dtype: DType, rank: Int
    ](
        output: OutputTensor[dtype=dtype, rank=rank, ...],
        inputs: StableInputVariadicTensors[dtype=dtype, rank=rank, ...],
        builder: DeviceGraphBuilder,
    ):
        pass


# Single-argument sibling of `imposter_stable_build_graph`: `build_graph`
# takes one `StableInputTensor` where `execute` takes an `InputTensor`.
@extensibility.register("imposter_stable_single_build_graph")
struct StableSingleBuildGraph:
    @staticmethod
    def execute[
        dtype: DType, rank: Int
    ](
        output: OutputTensor[dtype=dtype, rank=rank, ...],
        x: InputTensor[dtype=dtype, rank=rank, ...],
        ctx: DeviceContext,
    ):
        pass

    @staticmethod
    def build_graph[
        dtype: DType, rank: Int
    ](
        output: OutputTensor[dtype=dtype, rank=rank, ...],
        x: StableInputTensor[dtype=dtype, rank=rank, ...],
        builder: DeviceGraphBuilder,
    ):
        pass


@extensibility.register("device_graph_fill")
struct DeviceGraphFill:
    """Device-graph test kernel that fills its output with the constant `7`.

    Unlike `imposter_build_graph_capable` (a no-op stub used for emit tests),
    `build_graph` here records a *real* compiled fill kernel via
    `DeviceGraphBuilder.add_function`, so a replayed graph produces a
    deterministic output that an end-to-end execution test can check. `execute`
    performs the equivalent fill on the direct (non-graph) path.
    """

    comptime _fill_value = 7

    @staticmethod
    def execute[
        dtype: DType, rank: Int
    ](
        output: OutputTensor[dtype=dtype, rank=rank, ...],
        x: InputTensor[dtype=dtype, rank=rank, ...],
        ctx: DeviceContext,
    ) raises:
        @inline(.always)
        def func[width: Int, alignment: Int = 1](idx: Coord) {var output}:
            output._fused_store[element_alignment=alignment](
                idx, SIMD[dtype, width](Self._fill_value)
            )

        elementwise[1, target="gpu"](func, output.shape_coord(), ctx)

    @staticmethod
    def build_graph[
        dtype: DType, rank: Int
    ](
        output: OutputTensor[dtype=dtype, rank=rank, ...],
        x: InputTensor[dtype=dtype, rank=rank, ...],
        builder: DeviceGraphBuilder,
    ) raises:
        var ptr = output.unsafe_ptr()
        var length = Int32(output.size())

        # Record a real compiled kernel that fills every output element with the
        # constant. The closure captures the output pointer and length by value;
        # `add_function` compiles it against the builder's device context and
        # adds it as a graph node, so replaying the graph re-runs this fill.
        def fill_kernel() {var}:
            var n = Int(length)
            var tid = global_idx.x
            if tid >= n:
                return
            ptr[unsafe_offset=tid] = Scalar[dtype](Self._fill_value)

        comptime block_dim = 256
        _ = builder.add_function(
            fill_kernel,
            grid_dim=ceildiv(Int(length), block_dim),
            block_dim=block_dim,
        )


@extensibility.register("device_graph_fill_from_host")
struct DeviceGraphFillFromHost:
    """Device-graph test kernel that fills its output with the value of a
    host-resident scalar input, read on the host at enqueue/record time.

    Models dispatch metadata (attention cache lengths and friends): the value
    never reaches the device as data -- it is read while enqueuing and baked
    into the recorded kernel, which is why host inputs key the device-graph
    cache by contents. Under recording, `h` is the graph's stable host twin
    that `allocate_stable` pre-filled with the caller's bytes. `execute`
    performs the same host read on the direct (non-graph) path.
    """

    @staticmethod
    def fill_kernel[
        dtype: DType
    ](
        ptr: Pointer[Scalar[dtype], MutAnyOrigin],
        value: Scalar[dtype],
        length: Int32,
    ):
        var n = Int(length)
        var tid = global_idx.x
        if tid >= n:
            return
        ptr[unsafe_offset=tid] = value

    @staticmethod
    def execute[
        dtype: DType, rank: Int
    ](
        output: OutputTensor[dtype=dtype, rank=rank, ...],
        h: Scalar[dtype=dtype],
        ctx: DeviceContext,
    ) raises:
        var ptr = output.unsafe_ptr()
        var length = Int32(output.size())

        comptime block_dim = 256
        ctx.enqueue_function[Self.fill_kernel[dtype]](
            ptr,
            h,
            length,
            grid_dim=ceildiv(length, block_dim),
            block_dim=block_dim,
        )

    @staticmethod
    def build_graph[
        dtype: DType, rank: Int
    ](
        output: OutputTensor[dtype=dtype, rank=rank, ...],
        h: Scalar[dtype=dtype],
        builder: DeviceGraphBuilder,
    ) raises:
        # The enqueue-time host read: the value is frozen into the recorded
        # kernel here, so replay soundness rests on the contents being part of
        # the graph cache key.
        var ptr = output.unsafe_ptr()
        var length = Int32(output.size())

        comptime block_dim = 256
        _ = builder.add_function[Self.fill_kernel[dtype]](
            ptr,
            h,
            length,
            grid_dim=ceildiv(length, block_dim),
            block_dim=block_dim,
        )


@extensibility.register("device_graph_increment")
struct DeviceGraphIncrement:
    """Device-graph test kernel that adds the constant `1` to every element.

    The end-to-end device-graph test chains this after `device_graph_fill`, so
    replaying the recorded graph computes `fill(7) + 1 == 8` for every element.
    Like `device_graph_fill`, `build_graph` records a *real* compiled kernel via
    `DeviceGraphBuilder.add_function` and `execute` performs the equivalent
    increment on the direct (non-graph) path.
    """

    comptime _increment = 1

    @staticmethod
    def execute[
        dtype: DType, rank: Int
    ](
        output: OutputTensor[dtype=dtype, rank=rank, ...],
        x: InputTensor[dtype=dtype, rank=rank, ...],
        ctx: DeviceContext,
    ) raises:
        @inline(.always)
        def func[
            width: Int, alignment: Int = 1
        ](idx: Coord) {var x, var output}:
            var val = rebind[SIMD[dtype, width]](x.load[width](idx)) + (
                Self._increment
            )
            output._fused_store[element_alignment=alignment](idx, val)

        elementwise[1, target="gpu"](func, output.shape_coord(), ctx)

    @staticmethod
    def build_graph[
        dtype: DType, rank: Int
    ](
        output: OutputTensor[dtype=dtype, rank=rank, ...],
        x: InputTensor[dtype=dtype, rank=rank, ...],
        builder: DeviceGraphBuilder,
    ) raises:
        var in_ptr = x.unsafe_ptr()
        var out_ptr = output.unsafe_ptr()
        var length = Int32(output.size())

        # Record a real compiled kernel that reads each input element, adds the
        # constant, and writes it to the output. The closure captures the input
        # and output pointers and the length by value; `add_function` compiles
        # it against the builder's device context and adds it as a graph node,
        # so replaying the graph re-runs this increment.
        def increment_kernel() {var}:
            var n = Int(length)
            var tid = global_idx.x
            if tid >= n:
                return
            out_ptr[unsafe_offset=tid] = in_ptr[unsafe_offset=tid] + Scalar[
                dtype
            ](Self._increment)

        comptime block_dim = 256
        _ = builder.add_function(
            increment_kernel,
            grid_dim=ceildiv(Int(length), block_dim),
            block_dim=block_dim,
        )


@extensibility.register("add_one", type="cpu")
struct AddOneCPU:
    @staticmethod
    def execute[
        target: StaticString,
    ](outp: OutputTensor, x: InputTensor, ctx: DeviceContext,) raises:
        print("AddOne CPU kernel")

        @inline(.always)
        def func[width: Int, alignment: Int = 1](idx: Coord) {var x, var outp}:
            var val = rebind[SIMD[outp.dtype, width]](x.load[width](idx)) + 1
            outp._fused_store[element_alignment=alignment](idx, val)

        elementwise[1, target="cpu"](func, outp.shape_coord(), ctx)


@extensibility.register("gpu_only_kernel", type="gpu")
struct GpuOnlyKernel:
    @staticmethod
    def execute[
        target: StaticString,
    ](outp: OutputTensor, x: InputTensor, ctx: DeviceContext,) raises:
        pass


@extensibility.register("add_one", type="gpu")
struct AddOneGPU:
    @staticmethod
    def execute[
        target: StaticString,
    ](outp: OutputTensor, x: InputTensor, ctx: DeviceContext,) raises:
        print("AddOne GPU kernel")

        @inline(.always)
        def func[width: Int, alignment: Int = 1](idx: Coord) {var x, var outp}:
            var val = rebind[SIMD[outp.dtype, width]](x.load[width](idx)) + 1
            outp._fused_store[element_alignment=alignment](idx, val)

        elementwise[1, target="gpu"](func, outp.shape_coord(), ctx)


# Mixed-lane e2e proof: a trait-bound dense output alongside two legacy
# `InputTensor` operands in the same `execute` signature, computing
# `z[i] = x[i] + y[i]`. Confirms the trait-lane and legacy-lane capture
# assembly compose in the same `CallExecuteOp::emitMojo` call.
@extensibility.register("trait_mixed_legacy_add")
struct TraitMixedLegacyAdd:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        z: Some[Output & TileTensorable],
        x: InputTensor,
        y: InputTensor,
        ctx: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(z), TileTensorable)
        var zt = z.to_tile_tensor()
        var n = x.size()
        var zf = zt.reshape(row_major(n))
        for i in range(n):
            var xv = x[i]
            var yv = rebind[Scalar[type_of(x).dtype]](y[i])
            zf.store((i,), rebind[SIMD[type_of(zt).dtype, 1]](xv + yv))


# Mirror of `trait_mixed_legacy_add` with the lanes swapped: a legacy
# `OutputTensor` alongside two trait-bound dense inputs, so the legacy write
# path (`OutputTensor.__setitem__`) and the trait read path
# (`.to_tile_tensor().load`) both fire in the same emitted kernel.
@extensibility.register("trait_mixed_legacy_out_add")
struct TraitMixedLegacyOutAdd:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        z: OutputTensor,
        x: Some[Input & TileTensorable],
        y: Some[Input & TileTensorable],
        ctx: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(x), TileTensorable)
        comptime assert conforms_to(type_of(y), TileTensorable)
        var xt = x.to_tile_tensor()
        var yt = y.to_tile_tensor[type_of(x).dtype]()
        var n = Int(x.shape().product())
        var xf = xt.reshape(row_major(n))
        var yf = yt.reshape(row_major(n))
        for i in range(n):
            var xv = xf.load[width=1]((i,))
            var yv = yf.load[width=1]((i,))
            z[i] = rebind[Scalar[z.dtype]](xv + yv)


# One-value-two-lanes proof: a trait-bound dense input NEXT TO a legacy
# `InputTensor` input, computing `z[i] = x[i] + y[i]`. Called with the same
# graph value on both input positions, `SimplifyKernelOperands` merges the
# duplicates into one exec-body argument that must then materialize one
# representation per lane (the GEX-4071 per-use init split) — one
# `mogg.tensor_arg.__init__` for the trait position and one
# `mogg.tensor.__init__` for the legacy position, over the same buffer.
@extensibility.register("trait_dual_lane_add")
struct TraitDualLaneAdd:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        z: OutputTensor,
        x: Some[Input & TileTensorable],
        y: InputTensor,
        ctx: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(x), TileTensorable)
        var xt = x.to_tile_tensor()
        var n = y.size()
        var xf = xt.reshape(row_major(n))
        for i in range(n):
            var xv = xf.load[width=1]((i,))
            var yv = rebind[SIMD[type_of(xt).dtype, 1]](y[i])
            z[i] = rebind[Scalar[z.dtype]](xv + yv)


# ===----------------------------------------------------------------------=== #
# Signature-only opaque kernels for exercising MOToMAP lowering coverage.
#
# The bodies are empty: MOToMAP reads only the argument roles (fused vs plain
# output) to decide which results get a store epilogue, so these kernels cover
# one/two outputs across all-fused, none-fused and partially-fused shapes.
# ===----------------------------------------------------------------------=== #


@extensibility.register("map_lower_one_fused_output")
struct MapLowerOneFusedOutput:
    @staticmethod
    def execute[
        target: StaticString,
        has_epilogue_fusion: Bool,
    ](res: FusedOutputTensor, a: InputTensor, ctx: DeviceContext,) raises:
        pass


@extensibility.register("map_lower_one_plain_output")
struct MapLowerOnePlainOutput:
    @staticmethod
    def execute[
        target: StaticString,
    ](res: OutputTensor, a: InputTensor, ctx: DeviceContext,) raises:
        pass


@extensibility.register("map_lower_no_context")
struct MapLowerNoContext:
    @staticmethod
    def execute[
        target: StaticString,
    ](res: OutputTensor, a: InputTensor,) raises:
        pass


# What a declarative kernel's `execute_configuration` returns: the graph
# compiler calls only `get_grid_dim()` and `get_block_dim()`.
@fieldwise_init
struct DeclarativeLaunchConfig(Copyable):
    var grid_dim: Dim
    var block_dim: Dim

    def get_grid_dim(self) -> Dim:
        return self.grid_dim

    def get_block_dim(self) -> Dim:
        return self.block_dim


# A declarative kernel: `execute_device` is the device code and
# `execute_configuration`, which has a defaulted parameter, its host-side launch
# configuration. The graph compiler writes the launch.
@extensibility.register("map_lower_declarative")
struct MapLowerDeclarative:
    @staticmethod
    def execute_configuration[
        block: Int = 4
    ](res: Some[Tensor], a: Some[Tensor]) -> DeclarativeLaunchConfig:
        return DeclarativeLaunchConfig(grid_dim=Dim(1), block_dim=Dim(block))

    @staticmethod
    def execute_device(
        res: Some[Output & TileTensorable], a: Some[Input & TileTensorable]
    ):
        pass


# A signature-only in-place kernel with a mutable destination and a plain
# (non-fused) immutable input, mirroring `InplaceMemcpy`. Used to check that
# `FuseMutableLoads` fusing a `mo.mutable.load` into the immutable `src` operand
# still lowers to an immutable tensor view rather than a mutable buffer.
@extensibility.register("map_lower_inplace_copy")
struct MapLowerInplaceCopy:
    @staticmethod
    def execute[
        target: StaticString,
    ](dst: MutableInputTensor, src: InputTensor, ctx: DeviceContext,) raises:
        pass


@extensibility.register("map_lower_two_fused_outputs")
struct MapLowerTwoFusedOutputs:
    @staticmethod
    def execute[
        target: StaticString,
        has_epilogue_fusion: Bool,
    ](
        res0: FusedOutputTensor,
        res1: FusedOutputTensor,
        a: InputTensor,
        ctx: DeviceContext,
    ) raises:
        pass


# A variadic FUSED output: its results pack into a `_FusedOutputVariadicTensors`
# (the `fused_output` variadic ABI). Signature-only, like the other
# `map_lower_*` stubs; driven by a multi-result `mo.custom`.
@extensibility.register("map_lower_variadic_fused_output")
struct MapLowerVariadicFusedOutput:
    @staticmethod
    def execute[
        dtype: DType,
        rank: Int,
        size: Int,
        target: StaticString,
        has_epilogue_fusion: Bool,
    ](
        outputs: FusedOutputVariadicTensors[
            dtype=dtype, rank=rank, size=size, ...
        ],
        a: InputTensor,
        ctx: DeviceContext,
    ) raises:
        pass


@extensibility.register("map_lower_two_plain_outputs")
struct MapLowerTwoPlainOutputs:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        res0: OutputTensor,
        res1: OutputTensor,
        a: InputTensor,
        ctx: DeviceContext,
    ) raises:
        pass


@extensibility.register("map_lower_partial_fused_outputs")
struct MapLowerPartialFusedOutputs:
    @staticmethod
    def execute[
        target: StaticString,
        has_epilogue_fusion: Bool,
    ](
        res0: FusedOutputTensor,
        res1: OutputTensor,
        a: InputTensor,
        ctx: DeviceContext,
    ) raises:
        pass


# Two plain inputs feeding a single fused output: exercises the opaque body's
# output block arg (whose arg number is offset by the two leading input block
# args) resolving to the right captured tensor.
@extensibility.register("map_lower_two_inputs_one_fused_output")
struct MapLowerTwoInputsOneFusedOutput:
    @staticmethod
    def execute[
        target: StaticString,
        has_epilogue_fusion: Bool,
    ](
        res: FusedOutputTensor,
        a: InputTensor,
        b: InputTensor,
        ctx: DeviceContext,
    ) raises:
        pass


@extensibility.register("map_lower_scalar")
struct MapLowerScalar:
    @staticmethod
    def execute[
        dtype: DType,
        target: StaticString,
    ](
        res: OutputTensor,
        a: InputTensor,
        s: Scalar[dtype],
        ctx: DeviceContext,
    ) raises:
        pass


@extensibility.register("map_lower_variadic")
struct MapLowerVariadic:
    @staticmethod
    def execute[
        dtype: DType,
        size: Int,
        target: StaticString,
    ](
        res: OutputTensor,
        inputs: InputVariadicTensors[dtype=dtype, rank=1, size=size, ...],
        ctx: DeviceContext,
    ) raises:
        pass


# A real-body counterpart to `map_lower_one_plain_output`: same plain-output
# signature (so it lowers through the identical MAP plain path), but it actually
# writes `res = a + 1` so an end-to-end run can validate a numeric result.
@extensibility.register("map_lower_add_one")
struct MapLowerAddOne:
    @staticmethod
    def execute[
        target: StaticString,
    ](res: OutputTensor, a: InputTensor, ctx: DeviceContext,) raises:
        @inline(.always)
        def func[width: Int](idx: Coord) {var} -> SIMD[res.dtype, width]:
            return rebind[SIMD[res.dtype, width]](a.load[width](idx)) + 1

        foreach[target=target](func, res, ctx)


# Real-body plain-output kernels below broaden the advanced-fusion MAP-ABI
# execution coverage past the single 1-in/1-out shape. Each mirrors
# `map_lower_add_one`: a plain `OutputTensor` (no epilogue fusion) driven by a
# real `foreach` closure, so the MAP plain path both lowers AND computes a
# deterministic result an end-to-end run can validate. Output shape is
# identity-from-first-input for all of them, so none needs a shape function.


# 3 inputs / 1 output: `res = a + b + c`.
@extensibility.register("map_lower_add3")
struct MapLowerAdd3:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        res: OutputTensor,
        a: InputTensor,
        b: InputTensor,
        c: InputTensor,
        ctx: DeviceContext,
    ) raises:
        @inline(.always)
        def func[width: Int](idx: Coord) {var} -> SIMD[res.dtype, width]:
            return (
                rebind[SIMD[res.dtype, width]](a.load[width](idx))
                + rebind[SIMD[res.dtype, width]](b.load[width](idx))
                + rebind[SIMD[res.dtype, width]](c.load[width](idx))
            )

        foreach[target=target](func, res, ctx)


# 1 input / 2 outputs: `res0 = a + 1`, `res1 = a * 2` (two plain outputs, two
# independent `foreach` loops).
@extensibility.register("map_lower_two_outputs")
struct MapLowerTwoOutputs:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        res0: OutputTensor,
        res1: OutputTensor,
        a: InputTensor,
        ctx: DeviceContext,
    ) raises:
        @inline(.always)
        def add_one[width: Int](idx: Coord) {var} -> SIMD[res0.dtype, width]:
            return rebind[SIMD[res0.dtype, width]](a.load[width](idx)) + 1

        foreach(add_one, res0, ctx)

        @inline(.always)
        def times_two[width: Int](idx: Coord) {var} -> SIMD[res1.dtype, width]:
            return rebind[SIMD[res1.dtype, width]](a.load[width](idx)) * 2

        foreach(times_two, res1, ctx)


# 2 inputs / 2 outputs: `res0 = a + b`, `res1 = a - b`.
@extensibility.register("map_lower_add_sub")
struct MapLowerAddSub:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        res0: OutputTensor,
        res1: OutputTensor,
        a: InputTensor,
        b: InputTensor,
        ctx: DeviceContext,
    ) raises:
        @inline(.always)
        def add[width: Int](idx: Coord) {var} -> SIMD[res0.dtype, width]:
            return rebind[SIMD[res0.dtype, width]](a.load[width](idx)) + rebind[
                SIMD[res0.dtype, width]
            ](b.load[width](idx))

        foreach[target=target](add, res0, ctx)

        @inline(.always)
        def sub[width: Int](idx: Coord) {var} -> SIMD[res1.dtype, width]:
            return rebind[SIMD[res1.dtype, width]](a.load[width](idx)) - rebind[
                SIMD[res1.dtype, width]
            ](b.load[width](idx))

        foreach[target=target](sub, res1, ctx)


# Scalar operand: `res = a * s` where `s: Scalar[dtype]`. Exercises the MAP
# `scalar.create` materialization path with a real computed result.
@extensibility.register("map_lower_scale")
struct MapLowerScale:
    @staticmethod
    def execute[
        dtype: DType,
        target: StaticString,
    ](
        res: OutputTensor[dtype=dtype, ...],
        a: InputTensor[dtype=dtype, ...],
        s: Scalar[dtype],
        ctx: DeviceContext,
    ) raises:
        @inline(.always)
        def func[width: Int](idx: Coord) {var} -> SIMD[res.dtype, width]:
            return rebind[SIMD[res.dtype, width]](a.load[width](idx)) * rebind[
                Scalar[res.dtype]
            ](s)

        foreach(func, res, ctx)


# Variadic operand: `res = sum(inputs)` over an `InputVariadicTensors`.
# Exercises the MAP `tuple.create` materialization path. Modeled on the
# signature-only `map_lower_variadic` but with a real summing body.
@extensibility.register("map_lower_var_sum")
struct MapLowerVarSum:
    @staticmethod
    def execute[
        dtype: DType,
        size: Int,
        target: StaticString,
    ](
        res: OutputTensor[dtype=dtype, rank=1, ...],
        inputs: InputVariadicTensors[dtype=dtype, rank=1, size=size, ...],
        ctx: DeviceContext,
    ) raises:
        @inline(.always)
        def func[width: Int](idx: Coord) {var} -> SIMD[res.dtype, width]:
            var acc = SIMD[res.dtype, width](0)

            comptime for i in range(size):
                acc += rebind[SIMD[res.dtype, width]](
                    inputs[i].load[width](idx)
                )

            return acc

        foreach(func, res, ctx)


@extensibility.register("map_lower_two_fused_inputs")
struct MapLowerTwoFusedInputs:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        res: OutputTensor,
        a: FusedInputTensor,
        b: FusedInputTensor,
        ctx: DeviceContext,
    ) raises:
        pass


@extensibility.register("map_lower_partial_fused_inputs")
struct MapLowerPartialFusedInputs:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        res: OutputTensor,
        a: FusedInputTensor,
        b: InputTensor,
        ctx: DeviceContext,
    ) raises:
        pass


# `trait_tensor_tuple_fused_concat` through a host-built unified closure: the
# host projects the whole tuple to its access-enabled views with
# `TensorTuple._fused_view(inputs)`, the device closure copy-captures the view
# tuple, and the body loads element `k` through it. The view tuple is
# `RegisterPassable` only conditionally, which the closure emitter proves from
# the tuple's bindings. GPU only.
@extensibility.register("trait_tensor_tuple_fused_concat_unified_closure")
struct TraitTensorTupleFusedConcatUnifiedClosure:
    @staticmethod
    def execute[
        Ins: TypeList[Trait=Input & Fused & LoadAccess, ...],
        //,
        target: StaticString,
    ](
        z: Some[Output & TileTensorable],
        inputs: TensorTuple[*Ins],
        ctx: DeviceContext,
    ) raises:
        comptime if is_cpu[target]():
            raise Error(
                "trait_tensor_tuple_fused_concat_unified_closure only runs on"
                " the GPU"
            )
        else:
            comptime assert conforms_to(type_of(z), TileTensorable)
            var zt = z.to_tile_tensor()
            comptime out_dtype = type_of(zt).dtype
            var offsets = StaticTuple[Int32, len(Ins) + 1](fill=0)
            comptime for k in range(len(Ins)):
                offsets[k + 1] = offsets[k] + Int32(
                    Int(inputs[k].shape().product())
                )
            var n_dev = offsets[len(Ins)]
            var views = TensorTuple._fused_view(inputs)

            def device_body() {var views, var zt, var offsets, var n_dev}:
                var i = Int32(global_idx.x)
                if i >= n_dev:
                    return
                comptime for k in range(len(Ins)):
                    if i >= offsets[k] and i < offsets[k + 1]:
                        comptime assert conforms_to(type_of(views[k]), Loadable)
                        zt.reshape(row_major(Int(n_dev))).store(
                            (Int(i),),
                            views[k]
                            .load[1]((Int(i - offsets[k]),))
                            .cast[out_dtype](),
                        )

            comptime block_size = 64
            ctx.enqueue_function(
                device_body,
                grid_dim=(ceildiv(Int(n_dev), block_size),),
                block_dim=(block_size,),
            )


# A kernel with two `execute` overloads under one registration
# (`supports_unfused_execute`). The graph compiler calls the fused one when
# any output has a fused epilogue, with an identity epilogue on the others, and
# the plain one when none has, so the kernel keeps a direct store path without
# branching on `has_epilogue_fusion`. The plain overload needs its `where`
# clause: a fused output wrapper is a `Tensor` too, and Mojo does not rank the
# fused overload's bound as more specific. `z` is `x + 1` through either
# overload; `w` records which ran: 1 fused, 0 plain.
@extensibility.register("trait_unfused_execute")
struct TraitUnfusedExecute:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        z: Some[Output & Fused & StoreAccess],
        w: Some[Output & Fused & StoreAccess],
        x: Some[Input & Tensor],
        ctx: DeviceContext,
    ) raises:
        comptime assert (
            type_of(x).rank == 1
            and type_of(z).rank == 1
            and type_of(w).rank == 1
        )
        comptime if is_cpu[target]():
            comptime assert conforms_to(type_of(x), TileTensorable)
            comptime assert conforms_to(type_of(z), Storable)
            comptime assert conforms_to(type_of(w), Storable)
            var xt = x.to_tile_tensor()
            for i in range(Int(xt.dim[0]())):
                var raw = rebind[SIMD[type_of(z).dtype, 1]](xt.load((i,))) + 1
                z.store[1](Coord(i), raw)
                w.store[1](Coord(i), SIMD[type_of(w).dtype, 1](1))
        else:
            raise Error("trait_unfused_execute is CPU-only")

    @staticmethod
    def execute[
        target: StaticString,
    ](
        z: Some[Output & Tensor],
        w: Some[Output & Tensor],
        x: Some[Input & Tensor],
        ctx: DeviceContext,
    ) raises where not conforms_to(type_of(z), Fused):
        comptime assert (
            type_of(x).rank == 1
            and type_of(z).rank == 1
            and type_of(w).rank == 1
        )
        comptime if is_cpu[target]():
            comptime assert conforms_to(type_of(x), TileTensorable)
            comptime assert conforms_to(type_of(z), TileTensorable)
            comptime assert conforms_to(type_of(w), TileTensorable)
            var xt = x.to_tile_tensor()
            var zt = z.to_tile_tensor()
            var wt = w.to_tile_tensor()
            for i in range(Int(xt.dim[0]())):
                var raw = rebind[SIMD[type_of(zt).dtype, 1]](xt.load((i,))) + 1
                zt.store((i,), raw)
                wt.store((i,), SIMD[type_of(wt).dtype, 1](0))
        else:
            raise Error("trait_unfused_execute is CPU-only")
