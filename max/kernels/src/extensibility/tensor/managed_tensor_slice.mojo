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
"""
Implements the `ManagedTensorSlice` type - a view of a tensor that doesn't own
the underlying data. This type is used to build custom graph operations.
"""
from collections import Optional
from math import ceil, fma
from sys import align_of, simd_width_of, size_of
from sys.info import CompilationTarget, is_gpu
from sys.intrinsics import strided_load, strided_store

import algorithm
from buffer.dimlist import DimList, Dim, _make_partially_static_index_list
from builtin.device_passable import DevicePassable
from compiler_internal.directives import StaticTensorSpec, __mogg_intrinsic_attr
from gpu.host import get_gpu_target
from gpu.host.info import is_cpu
from gpu.host.info import is_gpu as _is_gpu
from layout import LayoutTensor
from layout._coord import Coord, _DimsToCoordLike
from layout._layout import Layout as TileLayout
from layout._tile_tensor import TileTensor
from memory import LegacyUnsafePointer

comptime UnsafePointer = LegacyUnsafePointer[mut=True, ...]
comptime OpaquePointer = LegacyUnsafePointer[
    mut=True, NoneType, origin=MutAnyOrigin
]
from register import register_internal
from runtime.asyncrt import DeviceContextPtr
from runtime.tracing import trace_arg
from tensor import RuntimeTensorSpec

from utils import IndexList, StaticTuple
from utils._serialize import _serialize

from ._indexing import _dot_prod, _row_major_strides, _slice_to_tuple
from .io_spec import IO, IOSpec

# ===----------------------------------------------------------------------=== #
# Load / Store Helper primitives
# ===----------------------------------------------------------------------=== #


@parameter
@always_inline
fn _gcd_pow2[a: Int, b: Int]() -> Int:
    # alignments should always be powers of 2
    comptime assert (
        a.is_power_of_two() and b.is_power_of_two()
    ), "a and b must be powers of 2"
    return min(a, b)


# TODO(GEX-1523): Consider moving these and other methods implementation into
# non-class member functions.
#
# TODO(GEX-1831): Remove redundant parameters present in the StaticTensorSpec
#
# Note: these methods are forced inline in the graph compiler. We keep the
# inlining at the whims of the automatic inliner for now since we want to
# predictably introspect and manipulate these particular functions.
#
# They are set to be inlined further down graph compiler stack.
@doc_private
@register_internal("simd_store_into_managed_tensor_slice")
@always_inline
fn simd_store_into_managed_tensor_slice[
    dtype: DType,
    rank: Int,
    simd_width: Int,
    static_spec: StaticTensorSpec[dtype, rank],
    element_alignment: Int = 1,
](
    tensor: ManagedTensorSlice[static_spec=static_spec],
    indices: IndexList[rank],
    value: SIMD[dtype, simd_width],
):
    var flat_index = tensor._compute_offset(indices)

    # Store alignment cannot exceed the data type's alignment.
    comptime max_alignment = _gcd_pow2[
        tensor.alignment, element_alignment * align_of[dtype]()
    ]()

    comptime static_stride = tensor._static_strides.at[rank - 1]()

    # Stride = 1
    @parameter
    @always_inline
    fn store_stride1():
        comptime if dtype == DType.bool:
            var v = value.cast[DType.uint8]()
            tensor._ptr.bitcast[UInt8]().store(flat_index, v)
        else:
            tensor._ptr.store[alignment=max_alignment](flat_index, value)

    # Stride > 1
    @parameter
    @always_inline
    fn store_strided(stride: Int):
        comptime if dtype == DType.bool:
            var v = value.cast[DType.uint8]()
            strided_store(
                v,
                tensor._ptr.bitcast[UInt8]() + flat_index,
                stride,
            )
        else:
            return strided_store(value, tensor._ptr + flat_index, stride)

    comptime if static_stride.is_dynamic():
        var stride = tensor._runtime_strides[rank - 1]
        # Dynamic stride
        if stride == 0:
            tensor._ptr.store[alignment=max_alignment](0, value)
        elif stride == 1:
            store_stride1()
        else:
            store_strided(stride)
    else:
        # static stride
        comptime if static_stride.get() == 0:
            tensor._ptr.store[alignment=max_alignment](0, value)
        elif static_stride.get() == 1:
            store_stride1()
        else:
            store_strided(static_stride.get())


@doc_private
@register_internal("simd_store_into_tensor_pointer")
@always_inline
fn simd_store_into_tensor_pointer[
    dtype: DType,
    rank: Int,
    static_spec: StaticTensorSpec[dtype, rank],
    simd_width: Int,
    element_alignment: Int = 1,
](
    ptr: UnsafePointer[Scalar[dtype]],
    shape: IndexList[rank],
    strides: IndexList[rank],
    indices: IndexList[rank],
    value: SIMD[dtype, simd_width],
):
    """Store a SIMD vector to raw tensor components.

    This function is GPU-safe because it only takes trivial types (pointer,
    IndexList) that can be properly captured in GPU kernel closures. Use this
    instead of simd_store_into_managed_tensor_slice when generating code for
    GPU kernels.

    Parameters:
        dtype: The data type of tensor elements.
        rank: The rank (number of dimensions) of the tensor.
        static_spec: The static specs of the tensor.
        simd_width: The SIMD width for the load operation.
        element_alignment: The element alignment for the load.

    Args:
        ptr: The raw pointer to tensor data.
        shape: The runtime shape of the tensor.
        strides: The runtime strides of the tensor.
        indices: The indices to store into.
        value: The value to store.
    """
    var tensor = OutputTensor[dtype=dtype, rank=rank, static_spec=static_spec](
        ptr, shape, strides
    )
    simd_store_into_managed_tensor_slice[element_alignment=element_alignment](
        tensor, indices, value
    )


# GPU-safe load function that takes raw components (pointer, strides) instead of
# ManagedTensorSlice. This avoids capturing ManagedTensorSlice in GPU kernels,
# which doesn't work correctly due to closure capture limitations.
@doc_private
@register_internal("simd_load_from_tensor_pointer")
@always_inline
fn simd_load_from_tensor_pointer[
    dtype: DType,
    rank: Int,
    static_spec: StaticTensorSpec[dtype, rank],
    simd_width: Int,
    element_alignment: Int = 1,
](
    ptr: UnsafePointer[Scalar[dtype]],
    shape: IndexList[rank],
    strides: IndexList[rank],
    indices: IndexList[rank],
) -> SIMD[dtype, simd_width]:
    """Load a SIMD vector from raw tensor components.

    This function is GPU-safe because it only takes trivial types (pointer,
    IndexList) that can be properly captured in GPU kernel closures. Use this
    instead of simd_load_from_managed_tensor_slice when generating code for
    GPU kernels.

    Parameters:
        dtype: The data type of tensor elements.
        rank: The rank (number of dimensions) of the tensor.
        static_spec: The static specs of the tensor.
        simd_width: The SIMD width for the load operation.
        element_alignment: The element alignment for the load.

    Args:
        ptr: The raw pointer to tensor data.
        shape: The runtime shape of the tensor.
        strides: The runtime strides of the tensor.
        indices: The indices to load from.

    Returns:
        A SIMD vector with the loaded values.
    """
    var tensor = InputTensor[dtype=dtype, rank=rank, static_spec=static_spec](
        ptr, shape, strides
    )
    return simd_load_from_managed_tensor_slice[
        simd_width=simd_width, element_alignment=element_alignment
    ](tensor, indices)


@doc_private
@register_internal("simd_load_from_managed_tensor_slice")
@always_inline
fn simd_load_from_managed_tensor_slice[
    dtype: DType,
    rank: Int,
    simd_width: Int,
    static_spec: StaticTensorSpec[dtype, rank],
    element_alignment: Int = 1,
](
    tensor: ManagedTensorSlice[static_spec=static_spec],
    indices: IndexList[rank],
) -> SIMD[dtype, simd_width]:
    var flat_index = tensor._compute_offset(indices)
    comptime static_stride = tensor._static_strides.at[rank - 1]()

    # Load alignment cannot exceed the data type's alignment.
    comptime max_alignment = _gcd_pow2[
        tensor.alignment, element_alignment * align_of[dtype]()
    ]()
    comptime invariant = not tensor.io_spec.mut

    # Stride = 1
    @parameter
    @always_inline
    fn load_stride1() -> SIMD[dtype, simd_width]:
        comptime if dtype == DType.bool:
            var v = tensor._ptr.bitcast[UInt8]().load[
                width=simd_width,
                invariant=invariant,
            ](flat_index)
            return v.cast[dtype]()
        else:
            return tensor._ptr.load[
                width=simd_width, alignment=max_alignment, invariant=invariant
            ](flat_index)

    # Stride > 1
    @parameter
    @always_inline
    fn load_strided(stride: Int) -> SIMD[dtype, simd_width]:
        comptime if dtype == DType.bool:
            var v = strided_load[simd_width, invariant=invariant](
                tensor._ptr.bitcast[UInt8]() + flat_index,
                stride,
            )
            return v.cast[dtype]()
        else:
            return strided_load[simd_width, invariant=invariant](
                tensor._ptr + flat_index, stride
            )

    comptime if static_stride.is_dynamic():
        var stride = tensor._runtime_strides[rank - 1]
        # Dynamic stride
        if stride == 0:
            return tensor._ptr.load[invariant=invariant](flat_index)
        elif stride == 1:
            return load_stride1()
        else:
            return load_strided(stride)
    else:
        # Static stride
        comptime if static_stride.get() == 0:
            return tensor._ptr.load[invariant=invariant](flat_index)
        elif static_stride.get() == 1:
            return load_stride1()
        else:
            return load_strided(static_stride.get())


# ===----------------------------------------------------------------------=== #
# Input / output fusion primitives
# ===----------------------------------------------------------------------=== #


@no_inline
fn _extract_tensor_spec[
    dtype: DType,
    rank: Int,
    //,
    static_spec: StaticTensorSpec[dtype, rank],
]() -> type_of(static_spec):
    return static_spec


@no_inline
fn rebuild_static_tensor_specs_with_input_lambda[
    func_type: __TypeOfAllTypes,
    //,
    dtype: DType,
    rank: Int,
](
    spec: StaticTensorSpec[dtype, rank],
    in_lambda: func_type,
) -> StaticTensorSpec[dtype, rank]:
    return StaticTensorSpec[dtype, rank](
        shape=spec.shape,
        strides=spec.strides,
        alignment=spec.alignment,
        address_space=spec.address_space,
        exclusive=spec.exclusive,
        in_lambda=rebind[spec.in_lambda_t](in_lambda),
        out_lambda=None,
        out_compute_lambda=None,
    )


@no_inline
fn rebuild_static_tensor_specs_with_output_lambda[
    func_type: __TypeOfAllTypes,
    //,
    dtype: DType,
    rank: Int,
](
    spec: StaticTensorSpec[dtype, rank],
    out_lambda: func_type,
) -> StaticTensorSpec[dtype, rank]:
    return StaticTensorSpec[dtype, rank](
        shape=spec.shape,
        strides=spec.strides,
        alignment=spec.alignment,
        address_space=spec.address_space,
        exclusive=spec.exclusive,
        in_lambda=None,
        out_lambda=rebind[spec.out_lambda_t](out_lambda),
        out_compute_lambda=None,
    )


@no_inline
fn rebuild_static_tensor_specs_with_compute_output_lambda[
    func_type: __TypeOfAllTypes,
    //,
    dtype: DType,
    rank: Int,
](
    spec: StaticTensorSpec[dtype, rank],
    out_compute_lambda: func_type,
) -> StaticTensorSpec[dtype, rank]:
    return StaticTensorSpec[dtype, rank](
        shape=spec.shape,
        strides=spec.strides,
        alignment=spec.alignment,
        address_space=spec.address_space,
        exclusive=spec.exclusive,
        in_lambda=None,
        out_lambda=None,
        out_compute_lambda=rebind[spec.out_compute_lambda_t](
            out_compute_lambda
        ),
    )


# Helper function used in SliceMOGGDPSFunc to generate the body of the input lambda
@__mogg_intrinsic_attr("mogg.dps_input_fusion_hook")
@register_internal("mogg.dps_input_fusion_hook")
@no_inline
fn _input_fusion_hook_impl[
    mut: Bool,
    //,
    dtype: DType,
    rank: Int,
    io_spec: IOSpec[mut],
    static_spec: StaticTensorSpec[dtype, rank],
](
    tensor: ManagedTensorSlice[io_spec=io_spec, static_spec=static_spec]
) -> type_of(static_spec):
    @always_inline
    @parameter
    fn _input_lambda[
        _w: Int, _elem_align: Int = 1
    ](i: IndexList[rank]) -> SIMD[dtype, _w]:
        # We use these methods to help with fusion passes which manipulates
        # calls. It is helpful to have a registered function.
        return rebind[SIMD[dtype, _w]](
            simd_load_from_managed_tensor_slice[
                simd_width=_w, element_alignment=_elem_align
            ](tensor, i)
        )

    return _extract_tensor_spec[
        rebuild_static_tensor_specs_with_input_lambda[dtype, rank](
            static_spec,
            _input_lambda,
        )
    ]()


# Helper function used in SliceMOGGDPSFunc to generate the body of the output lambda
@__mogg_intrinsic_attr("mogg.dps_output_fusion_hook")
@register_internal("mogg.dps_output_fusion_hook")
@no_inline
fn _output_fusion_hook_impl[
    mut: Bool,
    //,
    dtype: DType,
    rank: Int,
    io_spec: IOSpec[mut],
    static_spec: StaticTensorSpec[dtype, rank],
](
    tensor: ManagedTensorSlice[io_spec=io_spec, static_spec=static_spec]
) -> type_of(static_spec):
    @always_inline
    @parameter
    fn _output_lambda[
        _w: Int, _elem_align: Int = 1
    ](i: IndexList[rank], v: SIMD[dtype, _w]):
        # We use these methods to help with fusion passes which manipulates
        # calls. It is helpful to have a registered function.
        simd_store_into_managed_tensor_slice[
            simd_width=_w,
            element_alignment=_elem_align,
        ](tensor, i, rebind[SIMD[dtype, _w]](v))

    return _extract_tensor_spec[
        rebuild_static_tensor_specs_with_output_lambda[dtype, rank](
            static_spec,
            _output_lambda,
        )
    ]()


# ===----------------------------------------------------------------------=== #
# Mixed precision output fusion hook used by experimental codegen path.
# ===----------------------------------------------------------------------=== #


@__mogg_intrinsic_attr("mogg.dps_mixed_precision_output_fusion_hook")
@register_internal("mogg.dps_mixed_precision_output_fusion_hook")
@no_inline
fn _mixed_precision_output_fusion_hook_impl[
    mut: Bool,
    //,
    # DType and rank after casting/view fusion.
    rank: Int,
    dst_dtype: DType,
    # DType, shape and strides before casting/view fusion.
    src_rank: Int,
    src_shape: DimList,
    src_strides: DimList,
    src_dtype: DType,
    io_spec: IOSpec[mut],
    static_spec: StaticTensorSpec[dst_dtype, rank],
](
    tensor: ManagedTensorSlice[io_spec=io_spec, static_spec=static_spec]
) -> StaticTensorSpec[src_dtype, src_rank]:
    @always_inline
    @parameter
    fn _output_lambda[
        _w: Int, _elem_align: Int = 1
    ](i: IndexList[src_rank], v: SIMD[src_dtype, _w]):
        # .... compiler-generated-code insert here!
        simd_store_into_managed_tensor_slice[
            simd_width=_w,
            element_alignment=_elem_align,
        ](tensor, rebind[IndexList[rank]](i), rebind[SIMD[dst_dtype, _w]](v))

    comptime mixed_in_spec = StaticTensorSpec[src_dtype, src_rank](
        shape=src_shape,
        strides=src_strides,
        alignment=static_spec.alignment,
        address_space=static_spec.address_space,
        exclusive=static_spec.exclusive,
        in_lambda=None,
        out_lambda=None,
        out_compute_lambda=None,
    )

    return _extract_tensor_spec[
        rebuild_static_tensor_specs_with_output_lambda[src_dtype, src_rank](
            mixed_in_spec,
            _output_lambda,
        )
    ]()


@__mogg_intrinsic_attr("mogg.dps_mixed_precision_compute_output_fusion_hook")
@register_internal("mogg.dps_mixed_precision_compute_output_fusion_hook")
@no_inline
fn _mixed_precision_compute_output_fusion_hook_impl[
    mut: Bool,
    //,
    # DType and rank after casting/view fusion.
    rank: Int,
    dst_dtype: DType,
    # DType and shape before casting/view fusion.
    src_rank: Int,
    src_shape: DimList,
    src_dtype: DType,
    io_spec: IOSpec[mut],
    static_spec: StaticTensorSpec[dst_dtype, rank],
](
    tensor: ManagedTensorSlice[io_spec=io_spec, static_spec=static_spec]
) -> StaticTensorSpec[src_dtype, src_rank]:
    @always_inline
    @parameter
    fn _compute_output_lambda[
        _w: Int, _elem_align: Int = 1
    ](i: IndexList[src_rank], v: SIMD[src_dtype, _w]) -> SIMD[src_dtype, _w]:
        return v

    comptime mixed_in_spec = StaticTensorSpec[src_dtype, src_rank](
        shape=src_shape,
        strides=static_spec.strides,
        alignment=static_spec.alignment,
        address_space=static_spec.address_space,
        exclusive=static_spec.exclusive,
        in_lambda=None,
        out_lambda=None,
        out_compute_lambda=None,
    )

    return _extract_tensor_spec[
        rebuild_static_tensor_specs_with_compute_output_lambda[
            src_dtype, src_rank
        ](
            mixed_in_spec,
            _compute_output_lambda,
        )
    ]()


@register_internal(
    "rebuild_mix_precision_static_tensor_specs_with_input_lambda"
)
@no_inline
fn rebuild_mix_precision_static_tensor_specs_with_input_lambda[
    func_type: __TypeOfAllTypes,
    //,
    src_dtype: DType,
    dst_dtype: DType,
    rank: Int,
](
    spec: StaticTensorSpec[src_dtype, rank],
    in_lambda: func_type,
    out result: StaticTensorSpec[dst_dtype, rank],
):
    return StaticTensorSpec[dst_dtype, rank](
        shape=spec.shape,
        strides=spec.strides,
        alignment=spec.alignment,
        address_space=spec.address_space,
        exclusive=spec.exclusive,
        in_lambda=rebind[result.in_lambda_t](in_lambda),
        out_lambda=None,
        out_compute_lambda=None,
    )


@__mogg_intrinsic_attr("mogg.dps_mixed_precision_input_fusion_hook")
@register_internal("mogg.dps_mixed_precision_input_fusion_hook")
@no_inline
fn _mixed_precision_input_fusion_hook_impl[
    mut: Bool,
    //,
    dst_dtype: DType,  # The DType after casting.
    src_dtype: DType,  # The DType before casting.
    rank: Int,
    io_spec: IOSpec[mut],
    static_spec: StaticTensorSpec[src_dtype, rank],
](
    tensor: ManagedTensorSlice[io_spec=io_spec, static_spec=static_spec]
) -> StaticTensorSpec[dst_dtype, rank]:
    @always_inline
    @parameter
    fn _input_lambda[
        _w: Int, _elem_align: Int = 1
    ](i: IndexList[rank]) -> SIMD[dst_dtype, _w]:
        # We use these methods to help with fusion passes which manipulates
        # calls. It is helpful to have a registered function.
        var v = rebind[SIMD[src_dtype, _w]](
            simd_load_from_managed_tensor_slice[
                simd_width=_w, element_alignment=_elem_align
            ](tensor, i)
        )
        # .... compiler-generated-code here to bridge between src and dst_dtype
        return rebind[SIMD[dst_dtype, _w]](v)

    return _extract_tensor_spec[
        rebuild_mix_precision_static_tensor_specs_with_input_lambda[
            src_dtype, dst_dtype, rank
        ](
            static_spec,
            _input_lambda,
        )
    ]()


# ===----------------------------------------------------------------------=== #
# ManagedTensorSlice class
# ===----------------------------------------------------------------------=== #

comptime OutputTensor = ManagedTensorSlice[io_spec=Output]
comptime InputTensor = ManagedTensorSlice[io_spec=Input]

comptime _MutableInputTensor = ManagedTensorSlice[io_spec=MutableInput]
comptime _FusedOutputTensor = ManagedTensorSlice[io_spec=FusedOutput]
comptime _FusedInputTensor = ManagedTensorSlice[io_spec=FusedInput]

comptime _FusedComputeOutputTensor = ManagedTensorSlice[
    io_spec=_FusedComputeOutput
]

comptime DynamicTensor[dtype: DType, rank: Int] = ManagedTensorSlice[
    io_spec=IOUnknown,
    static_spec = StaticTensorSpec[dtype, rank].create_unknown(),
]


@fieldwise_init
struct ManagedTensorSlice[
    mut: Bool,
    input: IO,
    dtype: DType,
    rank: Int,
    //,
    io_spec: IOSpec[mut, input],
    *,
    static_spec: StaticTensorSpec[dtype, rank],
](DevicePassable, Stringable, TrivialRegisterPassable, Writable):
    """A view of a tensor that does not own the underlying allocated pointer.
    When the object lifetime ends it does not free the underlying pointer.
    Conversely, if a `ManagedTensorSlice` is created, it will not extend the
    life of the underlying pointer.

    Therefore, the user must take care to keep the pointer alive until the last
    use of a `ManagedTensorSlice` instance. This class is useful for writing
    custom operations where memory is managed by an external runtime like in
    MAX's inference stack.
    """

    # `trait DevicePassable` implementation
    comptime device_type: AnyType = LayoutTensor[
        Self.dtype, Self.static_spec.to_layout(), MutAnyOrigin
    ]

    fn _to_device_type(self, target: MutOpaquePointer[_]):
        target.bitcast[Self.device_type]()[] = self.to_layout_tensor()

    @staticmethod
    fn get_type_name() -> String:
        return (
            "ManagedTensorSlice[mut = "
            + String(Self.mut)
            + ", dtype = "
            + String(Self.dtype)
            + ", rank = "
            + String(Self.rank)
            + ", static_spec (as Layout) = "
            + String(Self.static_spec.to_layout())
            + "]"
        )

    comptime address_space = Self.static_spec.address_space
    comptime alignment = Self.static_spec.alignment
    comptime exclusive = Self.static_spec.exclusive
    comptime _static_shape = Self.static_spec.shape
    comptime _static_strides = Self.static_spec.strides

    comptime _in_lambda = Self.static_spec.in_lambda
    comptime _out_lambda = Self.static_spec.out_lambda

    var _ptr: UnsafePointer[Scalar[Self.dtype]]
    var _spec: RuntimeTensorSpec[Self.dtype, Self.rank]
    var _runtime_strides: IndexList[Self.rank]

    fn __init__(
        out self,
        ptr: UnsafePointer[Scalar[Self.dtype]],
        slices: InlineArray[Slice, Self.rank],
        slicer_spec: RuntimeTensorSpec[Self.dtype, Self.rank],
    ):
        """Initializes a ManagedTensorSlice from a pointer, array of slices and
        tensor spec.

        In general, custom operations should not create `ManagedTensorSlice`
        instances, but instead use the ones provided by the MAX inference
        engine.
        """

        @parameter
        @always_inline
        fn start_fn(slice: Slice) -> Int:
            return slice.start.value()

        @parameter
        @always_inline
        fn stop_fn(slice: Slice) -> Int:
            return slice.end.value()

        @parameter
        @always_inline
        fn step_fn(slice: Slice) -> Int:
            return slice.step.or_else(1)

        var start = _slice_to_tuple[start_fn](slices)
        var stop = _slice_to_tuple[stop_fn](slices)
        var step = _slice_to_tuple[step_fn](slices)

        var adjusted_shape = IndexList[Self.rank]()
        for i in range(Self.rank):
            adjusted_shape[i] = Int(
                ceil(Float64(stop[i] - start[i]) / Float64(step[i]))
            )
        var slice_spec = RuntimeTensorSpec[Self.dtype](adjusted_shape)

        var slicer_strides = _row_major_strides(adjusted_shape)
        var start_offset = _dot_prod(start, slicer_strides)

        var strides = IndexList[Self.rank]()

        comptime for i in range(Self.rank):
            strides[i] = step[i] * slicer_strides[i]

        self = Self(ptr + start_offset, slice_spec, strides)

    fn __init__(
        out self,
        ptr: UnsafePointer[Scalar[Self.dtype]],
        shape: IndexList[Self.rank],
    ):
        """Initializes a ManagedTensorSlice from a pointer and shape.

        In general, custom operations should not create `ManagedTensorSlice`
        instances, but instead use the ones provided by the MAX inference
        engine.
        """
        self._ptr = ptr
        self._spec = RuntimeTensorSpec[Self.dtype, Self.rank](shape)
        self._runtime_strides = _row_major_strides(shape)

    fn __init__(
        out self,
        ptr: UnsafePointer[Scalar[Self.dtype]],
        shape: IndexList[Self.rank],
        strides: IndexList[Self.rank],
    ):
        """Initializes a ManagedTensorSlice from a pointer, shape, and strides.

        In general, custom operations should not create `ManagedTensorSlice`
        instances, but instead use the ones provided by the MAX inference
        engine.
        """
        self = Self(
            ptr,
            RuntimeTensorSpec[Self.dtype, Self.rank](shape),
            strides,
        )

    @always_inline
    fn __getitem__(self, indices: IndexList[Self.rank]) -> Scalar[Self.dtype]:
        """Gets the value at the specified indices.

        Args:
          indices: The indices of the value to retrieve.

        Returns:
          The value at the specified indices.
        """
        comptime assert (
            not Self.static_spec.in_lambda
        ), "Direct load on fused tensor is forbidden"
        var offset = _dot_prod(indices, self.strides())
        return self._ptr[offset]

    @always_inline
    fn __getitem__(self, *indices: Int) -> Scalar[Self.dtype]:
        """Gets the value at the specified indices.

        Args:
          indices: The indices of the value to retrieve.

        Returns:
          The value at the specified indices.
        """
        comptime assert (
            not Self.static_spec.in_lambda
        ), "Direct load on fused tensor is forbidden"
        debug_assert(
            len(indices) == Self.rank,
            "mismatch between requested index and rank",
        )
        return self[IndexList[Self.rank](indices)]

    @always_inline
    fn __setitem__(self, *indices: Int, val: Scalar[Self.dtype]):
        """Stores the value at the specified indices.

        Args:
          indices: The indices of the value to store.
          val: The value to store.

        """
        comptime assert (
            not Self.static_spec.out_lambda
        ), "Direct store on fused tensor is forbidden"
        debug_assert(
            len(indices) == Self.rank,
            "mismatch between requested index and rank",
        )
        self[IndexList[Self.rank](indices)] = val

    @always_inline
    fn __setitem__(
        self, indices: IndexList[Self.rank], val: Scalar[Self.dtype]
    ):
        """Stores the value at the specified indices.

        Args:
          indices: The indices of the value to store.
          val: The value to store.

        """
        comptime assert (
            not Self.static_spec.out_lambda
        ), "Direct store on fused tensor is forbidden"
        var offset = _dot_prod(indices, self.strides())
        self._ptr[offset] = val

    fn spec(self) -> RuntimeTensorSpec[Self.dtype, Self.rank]:
        """Gets the `TensorSpec` of this tensor slice, which provides meta-data
        about the tensor slice.

        Returns:
            The static `TensorSpec` for this tensor slice.
        """
        return self._spec

    @always_inline
    fn shape(self) -> IndexList[Self.rank]:
        """Gets the shape of this tensor slice, as an `IndexList`.

        Returns:
            The shape of this tensor slice.
        """
        return _make_partially_static_index_list[Self.rank, Self._static_shape](
            self._spec.shape
        )

    @always_inline
    fn dim_size(self, index: Int) -> Int:
        """Gets the size of a given dimension of this tensor slice using a run
        time value.

        Args:
            index: The zero-based index of the dimension.

        Returns:
            The size of the tensor slice in the given dimension.
        """
        return self.shape()[index]

    @always_inline
    fn dim_size[index: Int](self) -> Int:
        """Gets the size of a given dimension of this tensor slice using a
        compile time value.

        Parameters:
            index: The zero-based index of the dimension.

        Returns:
            The size of the tensor slice in the given dimension.
        """

        comptime if Self._static_shape.at[index]().is_dynamic():
            return self._spec.shape[index]
        else:
            return Self._static_shape.get[index]()

    @always_inline
    fn strides(self) -> IndexList[Self.rank]:
        """Gets the strides of this tensor slice, as an `IndexList`.

        Returns:
            The strides of this tensor slice.
        """
        return _make_partially_static_index_list[
            Self.rank, Self._static_strides
        ](self._runtime_strides)

    @always_inline
    fn stride_length(self, index: Int) -> Int:
        """Gets the length of the stride of a given dimension of this tensor
        slice using a run time value.

        Args:
            index: The zero-based index of the dimension.

        Returns:
            The size of the tensor slice in the given dimension.
        """
        return self.strides()[index]

    @always_inline
    fn stride_length[index: Int](self) -> Int:
        """Gets the length of the stride of a given dimension of this tensor
        slice using a compile time value.

        Parameters:
            index: The zero-based index of the dimension.

        Returns:
            The size of the tensor slice in the given dimension.
        """

        comptime if Self._static_strides.at[index]().is_dynamic():
            return self._runtime_strides[index]
        else:
            return Self._static_strides.get[index]()

    @always_inline
    fn size(self) -> Int:
        """Computes the tensor slice's number of elements.

        Returns:
            The total number of elements in the tensor slice.
        """
        var product: Int = 1

        comptime for i in range(Self.rank):
            product *= self.dim_size[i]()

        return product

    @always_inline
    fn unsafe_ptr[
        _dtype: DType = Self.dtype
    ](self) -> UnsafePointer[Scalar[_dtype]]:
        """Get the pointer stored in this tensor slice.

        Since this method obtains the pointer stored in this tensor slice, it
        can modify the invariants of this tensor slice and lead to unexpected
        behavior. It should be used with caution.

        Parameters:
            _dtype: The type of the `UnsafePointer` in this tensor slice.

        Returns:
            The `UnsafePointer` which contains the data for this tensor slice.
        """
        return rebind[UnsafePointer[Scalar[_dtype]]](self._ptr)

    @always_inline
    fn load[
        width: Int,
        # Necessary to make it simpler on the call site.
        _rank: Int,
        element_alignment: Int = 1,
    ](self, index: IndexList[_rank]) -> SIMD[Self.dtype, width]:
        """Gets data from this tensor slice as a `SIMD`.

        Parameters:
            width: The width of the `SIMD` value. This must be large enough to contain the data from this tensor slice.
            _rank: The rank of the tensor slice.
            element_alignment: Indicate the alignment of the pointer stored to memory. This is needed to issue vector load for GPUs with strict alignment requirements.

        Args:
            index: An `IndexList` of size `_rank` to indicate the dimension of the tensor slice to obtain data from.

        Returns:
            Data from this tensor slice at dimension `index`.
        """
        comptime assert (
            Self.input == IO.Input or Self.input == IO.Unknown
        ), "loading not supported for output tensors"

        comptime assert _rank == Self.rank
        var ridx = rebind[IndexList[Self.rank]](index)
        return simd_load_from_managed_tensor_slice[
            simd_width=width, element_alignment=element_alignment
        ](self, ridx)

    @__mogg_intrinsic_attr("mogg.tensor_fused_load")
    @always_inline
    fn _fused_load[
        width: Int,
        # Necessary to make it simpler on the call site.
        _rank: Int,
        element_alignment: Int = 1,
    ](self, index: IndexList[_rank]) capturing -> SIMD[Self.dtype, width]:
        comptime assert _rank == Self.rank
        var ridx = rebind[IndexList[Self.rank]](index)

        comptime in_lambda = Self.static_spec.in_lambda
        comptime alignment = Self.static_spec.alignment
        comptime address_space = Self.static_spec.address_space
        comptime strides = Self.static_spec.strides

        comptime if in_lambda:
            comptime in_fn = in_lambda.value()
            return in_fn[width, element_alignment](ridx)
        else:
            return simd_load_from_managed_tensor_slice[
                simd_width=width, element_alignment=element_alignment
            ](self, ridx)

    @always_inline("nodebug")
    fn _lambda_load[
        width: Int,
        # Necessary to make it simpler on the call site.
        _rank: Int,
        element_alignment: Int = 1,
    ](self, index: IndexList[_rank]) -> SIMD[Self.dtype, width]:
        comptime assert _rank == Self.rank
        var ridx = rebind[IndexList[Self.rank]](index)
        comptime in_lambda = Self.static_spec.in_lambda
        comptime assert Bool(in_lambda)
        comptime in_fn = in_lambda.value()
        return in_fn[width, element_alignment](ridx)

    @always_inline
    fn _compute_offset(self, index: IndexList[Self.rank]) -> Int:
        comptime if Self.rank == 0:
            return 0

        # Special case for NVidia GPU on shared memory.
        # We can do the offset computation in int32 instead.
        comptime if is_gpu() and Self.address_space in (
            AddressSpace.SHARED,
            AddressSpace.LOCAL,
            AddressSpace.CONSTANT,
        ):
            var offset: Int32 = 0

            comptime for i in range(Self.rank):
                comptime if Self._static_strides.at[i]().is_dynamic():
                    offset = fma(
                        Int32(index[i]), Int32(self._runtime_strides[i]), offset
                    )
                else:
                    offset = fma(
                        Int32(index[i]),
                        Int32(Self._static_strides.get[i]()),
                        offset,
                    )
            return Int(offset)

        var offset = 0

        comptime for i in range(Self.rank):
            comptime if Self._static_strides.at[i]().is_dynamic():
                offset = fma(index[i], self._runtime_strides[i], offset)
            else:
                offset = fma(index[i], Self._static_strides.get[i](), offset)

        return offset

    @always_inline
    fn store[
        width: Int,
        # Necessary to make it simpler on the call site.
        _rank: Int,
        element_alignment: Int = 1,
    ](
        self: ManagedTensorSlice[
            mut=True,
            static_spec = Self.static_spec,
        ],
        index: IndexList[_rank],
        val: SIMD[Self.dtype, width],
    ):
        """Sets data in this tensor slice from a `SIMD`.

        Parameters:
            width: The width of the `SIMD` value.
            _rank: The rank of the tensor slice.
            element_alignment: Indicate the alignment of the pointer stored to memory. This is needed to issue vector store for GPUs with strict alignment requirements.

        Args:
            index: An `IndexList` of size `_rank` to indicate the dimension of the tensor slice to set data in.
            val: The data to set into this tensor slice.
        """
        comptime assert _rank == Self.rank
        var ridx = rebind[IndexList[Self.rank]](index)

        simd_store_into_managed_tensor_slice[
            simd_width=width,
            element_alignment=element_alignment,
        ](self, ridx, val)

    @__mogg_intrinsic_attr("mogg.tensor_fused_store")
    @always_inline
    fn _fused_store[
        width: Int,
        # Necessary to make it simpler on the call site.
        _rank: Int,
        element_alignment: Int = 1,
    ](
        self: ManagedTensorSlice[mut=True, static_spec = Self.static_spec],
        index: IndexList[_rank],
        val: SIMD[Self.dtype, width],
    ) capturing:
        comptime assert _rank == Self.rank
        var ridx = rebind[IndexList[Self.rank]](index)

        comptime out_lambda = Self.static_spec.out_lambda
        comptime alignment = Self.static_spec.alignment
        comptime address_space = Self.static_spec.address_space
        comptime strides = Self.static_spec.strides

        comptime if out_lambda:
            comptime out_fn = out_lambda.value()
            out_fn[width, element_alignment](ridx, val)
        else:
            simd_store_into_managed_tensor_slice[
                simd_width=width,
                element_alignment=element_alignment,
            ](self, ridx, val)

    @always_inline("nodebug")
    fn _lambda_store[
        width: Int,
        # Necessary to make it simpler on the call site.
        _rank: Int,
        element_alignment: Int = 1,
    ](
        self: ManagedTensorSlice[
            io_spec = IOSpec[True, Self.input](),
            static_spec = Self.static_spec,
        ],
        index: IndexList[_rank],
        val: SIMD[Self.dtype, width],
    ):
        comptime assert _rank == Self.rank
        var ridx = rebind[IndexList[Self.rank]](index)
        comptime out_lambda = Self.static_spec.out_lambda
        comptime assert Bool(out_lambda)
        comptime out_fn = out_lambda.value()
        out_fn[width, element_alignment](ridx, val)

    @always_inline
    fn _fused_compute_output_lambda[
        width: Int,
        # Necessary to make it simpler on the call site.
        _rank: Int,
    ](
        self: ManagedTensorSlice[mut=True, static_spec = Self.static_spec],
        index: IndexList[_rank],
        val: SIMD[Self.dtype, width],
    ) capturing -> SIMD[Self.dtype, width]:
        comptime assert _rank == Self.rank
        var ridx = rebind[IndexList[Self.rank]](index)

        comptime out_compute_lambda = Self.static_spec.out_compute_lambda

        comptime if out_compute_lambda:
            comptime out_fn = out_compute_lambda.value()
            return out_fn[width](ridx, val)
        else:
            return val

    @always_inline
    fn with_layout[
        new_rank: Int,
        //,
        new_static_shape: DimList,
        new_static_strides: DimList,
    ](
        self,
        new_runtime_shape: IndexList[new_rank],
        new_runtime_strides: IndexList[new_rank],
        offset_ptr: Optional[UnsafePointer[Scalar[Self.dtype]]] = None,
        out result: ManagedTensorSlice[
            rank=new_rank,
            io_spec = Self.io_spec,
            static_spec = Self.static_spec.with_layout[new_rank](
                new_static_shape, new_static_strides
            ),
        ],
    ):
        comptime assert (
            len(new_static_shape) == new_rank
        ), "static shape has incorrect rank"
        comptime assert (
            len(new_static_strides) == new_rank
        ), "static strides has incorrect rank"
        debug_assert(
            _is_consistent[new_static_shape](new_runtime_shape)
            and _is_consistent[new_static_strides](new_runtime_strides)
        )

        return {
            offset_ptr.or_else(self._ptr),
            new_runtime_shape,
            new_runtime_strides,
        }

    @doc_private
    @always_inline
    fn _bind_to_fused_input[
        lambda_fn: StaticTensorSpec[Self.dtype, Self.rank].in_lambda_t
    ](
        self,
        out result: ManagedTensorSlice[
            dtype = Self.dtype,
            rank = Self.rank,
            io_spec=FusedInput,
            static_spec = rebuild_static_tensor_specs_with_input_lambda(
                Self.static_spec, lambda_fn
            ),
        ],
    ):
        constrained[
            not Self.static_spec.in_lambda
            and not Self.static_spec.out_lambda
            and not Self.static_spec.out_compute_lambda,
            "The tensor is already bound to a lambda",
        ]()
        return {self._ptr, self._spec, self._runtime_strides}

    @doc_private
    @always_inline
    fn _bind_to_fused_output[
        lambda_fn: StaticTensorSpec[Self.dtype, Self.rank].out_lambda_t
    ](
        self,
        out result: ManagedTensorSlice[
            dtype = Self.dtype,
            rank = Self.rank,
            io_spec=FusedOutput,
            static_spec = rebuild_static_tensor_specs_with_output_lambda(
                Self.static_spec, lambda_fn
            ),
        ],
    ):
        constrained[
            not Self.static_spec.in_lambda
            and not Self.static_spec.out_lambda
            and not Self.static_spec.out_compute_lambda,
            "The tensor is already bound to a lambda",
        ]()
        return {self._ptr, self._spec, self._runtime_strides}

    @doc_private
    @always_inline
    fn _bind_to_fused_compute_output[
        lambda_fn: StaticTensorSpec[Self.dtype, Self.rank].out_compute_lambda_t
    ](
        self,
        out result: ManagedTensorSlice[
            dtype = Self.dtype,
            rank = Self.rank,
            io_spec=_FusedComputeOutput,
            static_spec = rebuild_static_tensor_specs_with_compute_output_lambda(
                Self.static_spec, lambda_fn
            ),
        ],
    ):
        constrained[
            not Self.static_spec.in_lambda
            and not Self.static_spec.out_lambda
            and not Self.static_spec.out_compute_lambda,
            "The tensor is already bound to a lambda",
        ]()
        return {self._ptr, self._spec, self._runtime_strides}

    @doc_private
    @always_inline
    fn _bind_to_fused_compute_output[
        lambda_fn: StaticTensorSpec[Self.dtype, Self.rank].out_lambda_t
    ](
        self,
        out result: ManagedTensorSlice[
            dtype = Self.dtype,
            rank = Self.rank,
            io_spec=_FusedComputeOutput,
            static_spec = rebuild_static_tensor_specs_with_output_lambda(
                Self.static_spec, lambda_fn
            ),
        ],
    ):
        constrained[
            not Self.static_spec.in_lambda
            and not Self.static_spec.out_lambda
            and not Self.static_spec.out_compute_lambda,
            "The tensor is already bound to a lambda",
        ]()
        return {self._ptr, self._spec, self._runtime_strides}

    @always_inline
    fn to_layout_tensor(
        self,
        out result: LayoutTensor[
            Self.dtype, Self.static_spec.to_layout(), MutAnyOrigin
        ],
    ):
        comptime layout = Self.static_spec.to_layout()
        return type_of(result)(
            self.unsafe_ptr(),
            type_of(result.runtime_layout)(
                self.shape().cast[result.layout_int_type](),
                self.strides().cast[result.linear_idx_type](),
            ),
        )

    @always_inline
    fn to_tile_tensor[
        coord_dtype: DType
    ](
        self,
        out result: TileTensor[
            dtype = Self.dtype,
            origin=MutExternalOrigin,
            LayoutType = TileLayout[
                shape_types = _DimsToCoordLike[
                    coord_dtype, Self.static_spec.shape
                ],
                stride_types = _DimsToCoordLike[
                    coord_dtype, Self.static_spec.strides
                ],
            ],
        ],
    ):
        var shape_tuple = Coord[
            *_DimsToCoordLike[coord_dtype, Self.static_spec.shape]
        ]()
        var stride_tuple = Coord[
            *_DimsToCoordLike[coord_dtype, Self.static_spec.strides]
        ]()
        var shape = self.shape()
        var stride = self.strides()

        comptime for i in range(Self.rank):
            comptime if not shape_tuple.element_types[i].is_static_value:
                shape_tuple[i] = rebind[shape_tuple.element_types[i]](
                    Scalar[coord_dtype](shape[i])
                )

            comptime if not stride_tuple.element_types[i].is_static_value:
                stride_tuple[i] = rebind[stride_tuple.element_types[i]](
                    Scalar[coord_dtype](stride[i])
                )

        return {
            self.unsafe_ptr().unsafe_origin_cast[MutExternalOrigin](),
            layout._layout.Layout(shape_tuple, stride_tuple),
        }

    fn write_to(self, mut writer: Some[Writer]):
        """
        Formats this buffer to the provided Writer.

        Args:
            writer: The object to write to.
        """
        writer.write("ManagedTensorSlice(")

        @parameter
        fn serialize[T: Writable](val: T):
            writer.write(val)

        var shape = List[Int]()
        for i in range(Self.rank):
            shape.append(self.shape()[i])

        # TODO(1937): make this work with all valid strides
        _serialize[serialize_fn=serialize, serialize_end_line=False](
            self._ptr, shape
        )

        writer.write("){")
        writer.write("static_shape = ", self._static_shape)
        writer.write(", static_strides = ", self._static_strides)
        writer.write(", dynamic_shape = ", self.shape())
        writer.write(", dynamic_strides = ", self.strides())
        writer.write(", alignment = ", self.alignment)
        writer.write(", address_space = ", self.address_space)
        writer.write("}")

    @no_inline
    fn __repr__(self) -> String:
        """Gets the buffer as a string.

        Returns:
          A compact string representation of the buffer.
        """
        return self.__str__()

    @no_inline
    fn __str__(self) -> String:
        """Gets the buffer as a string.

        Returns:
          A compact string of the buffer.
        """
        return String.write(self)


fn _is_consistent[static_info: DimList](runtime_info: IndexList) -> Bool:
    comptime if len(static_info) != runtime_info.size:
        return False

    comptime for i in range(runtime_info.size):
        comptime if not static_info.has_value[i]():
            continue

        if static_info.at[i]() != runtime_info[i]:
            return False

    return True


# TODO: Move to oss/modular/mojo/stdlib/stdlib/runtime/tracing.mojo and
# rename to trace_arg
@always_inline
fn trace_slice_arg(name: String, buf: ManagedTensorSlice) -> String:
    """Helper to stringify the type and shape of a kernel argument for tracing.

    Args:
        name: The name of the argument.
        buf: The tensor to trace.

    Returns:
        A string representation of the buffer with its shape and data type.
    """
    return trace_arg(name, buf._runtime_strides, buf.dtype)


# ===----------------------------------------------------------------------=== #
# VariadicTensors
# ===----------------------------------------------------------------------=== #

comptime InputVariadicTensors = VariadicTensors[io_spec=Input]
comptime OutputVariadicTensors = VariadicTensors[io_spec=Output]

comptime _MutableInputVariadicTensors = VariadicTensors[io_spec=MutableInput]
comptime _FusedInputVariadicTensors = VariadicTensors[io_spec=FusedInput]
comptime _FusedOutputVariadicTensors = VariadicTensors[io_spec=FusedOutput]


@fieldwise_init
struct VariadicTensors[
    mut: Bool,
    input: IO,
    //,
    dtype: DType,
    rank: Int,
    size: Int,
    io_spec: IOSpec[mut, input],
    *,
    static_specs: StaticTuple[StaticTensorSpec[dtype, rank], size],
](Sized, TrivialRegisterPassable):
    """A tuple-like container of tensors representing variadic arguments from
    the graph compiler."""

    var _tensors: StaticTuple[DynamicTensor[Self.dtype, Self.rank], Self.size]

    fn __init__(
        out self,
        ptrs: StaticTuple[UnsafePointer[Scalar[Self.dtype]], Self.size],
        shapes: StaticTuple[IndexList[Self.rank], Self.size],
    ):
        """Initialize the variadic tensor from tuples of pointers and shapes.

        This is a bulk initialization of the VariadicTensors value from an
        array of pointers and an array of runtime shapes. This allows the graph
        compiler to avoid generating code to construct DynamicTensor values
        directly.
        """

        self._tensors = {}

        for i in range(Self.size):
            var tensor = DynamicTensor[Self.dtype, Self.rank](
                ptrs[i], shapes[i]
            )
            self._tensors._unsafe_ref(i) = tensor

    fn __len__(self) -> Int:
        """Returns the number of variadic arguments in the pack.

        Returns:
            The number of variadic arguments.
        """
        return Self.size

    fn __getitem__[
        index: Int
    ](
        self,
        out result: ManagedTensorSlice[
            io_spec = Self.io_spec, static_spec = Self.static_specs[index]
        ],
    ):
        """Returns the tensor at the given position in the variadic argument
        argument pack.

        Parameters:
            index: The index into the variadic tensor arguments.

        Returns:
            The tensor at the specified index.
        """
        comptime assert index < Self.size
        var tensor = self._tensors[index]
        return {tensor._ptr, tensor._spec, tensor._runtime_strides}


# ===----------------------------------------------------------------------=== #
# ForEach / view copy primitives
# ===----------------------------------------------------------------------=== #


@doc_private
fn get_kernel_simd_width[dtype: DType, target: StaticString]() -> Int:
    """Get the simd width used in lambda functions.

    For non-simd arch like GPU, this is the width in terms of number of elements
    used per load/store instruction.
    """

    comptime if _is_gpu[target]():
        # We hardcode simd width to 16B for Nvidia GPUs but >= sm_100
        # arch support 32B load/store to global memory, see KERN-2037.
        comptime if CompilationTarget[get_gpu_target()]._is_arch["sm_100a"]():
            return 32 // size_of[dtype]()

        return simd_width_of[dtype, target = get_gpu_target()]()

    return simd_width_of[dtype]()


@__mogg_intrinsic_attr("mogg.for_each")
@__mogg_intrinsic_attr("mogg.elemwise_for_each")
@no_inline
fn foreach[
    dtype: DType,
    rank: Int,
    //,
    func: fn[width: Int, element_alignment: Int](
        IndexList[rank]
    ) capturing -> SIMD[dtype, width],
    *,
    target: StaticString = "cpu",
    simd_width: Int = get_kernel_simd_width[dtype, target](),
    _trace_name: StaticString = "mogg.for_each",
    use_blocking_impl: Bool = False,
](
    tensor: ManagedTensorSlice[mut=True, dtype=dtype, rank=rank],
    ctx: DeviceContextPtr = DeviceContextPtr(),
) raises:
    """Apply the function `func` to each element of the tensor slice.

    Parameters:
        dtype: The data type of the elements in the tensor slice.
        rank: The rank of the tensor slice.
        func: The function to apply to each element of the tensor slice.
        target: Indicates the type of the target device (e.g. "cpu", "gpu").
        simd_width: The SIMD width for the target (usually leave this as its default value).
        _trace_name: Name of the executed operation displayed in the trace_description.
        use_blocking_impl: If the impl should use this thread for doing the work.

    Args:
        tensor: The output tensor slice which receives the return values from `func`.
        ctx: The call context (forward this from the custom operation).
    """
    debug_assert(
        ctx._handle or is_cpu[target](),
        "Expecting non-null device ctx for GPU kernels",
    )

    @parameter
    @always_inline
    fn elementwise_fn_wrapper[
        width: Int,
        rank: Int,
        alignment: Int = 1,
    ](index: IndexList[rank]) capturing:
        var val = func[width, alignment](rebind[IndexList[tensor.rank]](index))
        tensor._fused_store[element_alignment=alignment](index, val)

    algorithm.functional.elementwise[
        elementwise_fn_wrapper,
        simd_width,
        use_blocking_impl=use_blocking_impl,
        target=target,
        _trace_description=_trace_name,
    ](tensor.shape(), ctx)


@__mogg_intrinsic_attr("mogg.for_each")
@__mogg_intrinsic_attr("mogg.for_each.out_func")
@no_inline
fn foreach[
    dtype: DType,
    rank: Int,
    //,
    func: fn[width: Int](IndexList[rank]) capturing -> SIMD[dtype, width],
    out_func: fn[width: Int](IndexList[rank]) capturing[_] -> None,
    *,
    target: StaticString = "cpu",
    simd_width: Int = get_kernel_simd_width[dtype, target](),
    _trace_name: StaticString = "mogg.for_each",
    use_blocking_impl: Bool = False,
](
    tensor: ManagedTensorSlice[dtype=dtype, rank=rank],
    ctx: DeviceContextPtr = DeviceContextPtr(),
) raises:
    """Apply the function `func` to each element of the tensor slice.

    Parameters:
        dtype: The data type of the elements in the tensor slice.
        rank: The rank of the tensor slice.
        func: The function to apply to each element of the tensor slice.
        out_func: The function to apply on each output element.
        target: Indicates the type of the target device (e.g. "cpu", "gpu").
        simd_width: The SIMD width for the target (usually leave this as its default value).
        _trace_name: Name of the executed operation displayed in the trace_description.
        use_blocking_impl: If the impl should use this thread for doing the work.

    Args:
        tensor: The input tensor slice which the consumed values.
        ctx: The call context (forward this from the custom operation).
    """
    debug_assert(
        ctx._handle or is_cpu[target](),
        "Expecting non-null device ctx for GPU kernels",
    )

    @parameter
    @always_inline
    fn out_func_shim[
        _width: Int, _rank: Int, _alignment: Int = 1
    ](index: IndexList[_rank]) capturing:
        idx = rebind[IndexList[rank]](index)
        out_func[_width](idx)

    algorithm.functional.elementwise[
        out_func_shim,
        simd_width,
        use_blocking_impl=use_blocking_impl,
        target=target,
        _trace_description=_trace_name,
    ](tensor.shape(), ctx)


fn foreach[
    dtype: DType,
    rank: Int,
    //,
    func: fn[width: Int](IndexList[rank]) capturing -> SIMD[dtype, width],
    *,
    target: StaticString = "cpu",
    simd_width: Int = get_kernel_simd_width[dtype, target](),
    _trace_name: StaticString = "mogg.for_each",
    use_blocking_impl: Bool = False,
](
    tensor: ManagedTensorSlice[mut=True, dtype=dtype, rank=rank],
    ctx: DeviceContextPtr = DeviceContextPtr(),
) raises:
    """Apply the function `func` to each element of the tensor slice.

    Parameters:
        dtype: The data type of the elements in the tensor slice.
        rank: The rank of the tensor slice.
        func: The function to apply to each element of the tensor slice.
        target: Indicates the type of the target device (e.g. "cpu", "gpu").
        simd_width: The SIMD width for the target (usually leave this as its default value).
        _trace_name: Name of the executed operation displayed in the trace_description.
        use_blocking_impl: If the impl should use this thread for doing the work.

    Args:
        tensor: The output tensor slice which receives the return values from `func`.
        ctx: The call context (forward this from the custom operation).
    """

    @parameter
    @always_inline
    fn func_shim[
        width: Int, element_alignment: Int
    ](index: IndexList[rank]) capturing -> SIMD[dtype, width]:
        return func[width](index)

    foreach[
        dtype=dtype,
        rank=rank,
        func=func_shim,
        target=target,
        simd_width=simd_width,
        _trace_name=_trace_name,
        use_blocking_impl=use_blocking_impl,
    ](tensor, ctx)


# TensorCopy intrinsic used by view kernels.
# z is a kernel output, and x a view of the input.
@__mogg_intrinsic_attr("mogg.view_materialize")
@doc_private
@no_inline
fn view_copy_impl[
    dtype: DType,
    rank: Int,
    spec: StaticTensorSpec[dtype, rank],
    //,
    *,
    target: StaticString,
    _trace_name: StaticString = "mogg.view_copy_impl",
    use_blocking_impl: Bool = False,
](
    z: ManagedTensorSlice[mut=True, dtype=dtype, rank=rank],
    x: ManagedTensorSlice[static_spec=spec],
    ctx: DeviceContextPtr,
) raises:
    comptime assert _compatible_with[
        x._static_shape, z._static_shape
    ](), "static shapes not compatible"
    debug_assert(x.shape() == z.shape(), "runtime shapes not compatible")

    @parameter
    @always_inline
    fn func[
        width: Int, element_alignment: Int
    ](idx: IndexList[z.rank]) -> SIMD[z.dtype, width]:
        return simd_load_from_managed_tensor_slice[
            simd_width=width, element_alignment=element_alignment
        ](x, idx)

    foreach[
        func,
        target=target,
        _trace_name=_trace_name,
        use_blocking_impl=use_blocking_impl,
    ](z, ctx)


fn _compatible_with[x: DimList, y: DimList]() -> Bool:
    comptime if len(x) != len(y):
        return False

    comptime for i in range(len(x)):
        if x.has_value[i]() and y.has_value[i]() and x.at[i]() != y.at[i]():
            return False

    return True
