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
from std.collections import Optional
from std.gpu.host import DeviceBuffer, DeviceContext
from std.math import ceil, fma
from std.sys import align_of, simd_width_of, size_of
from std.sys.info import CompilationTarget, is_gpu
from std.sys.intrinsics import _type_is_eq, strided_load, strided_store

import std.algorithm
from layout import CoordLike, IntTuple
from std.builtin.device_passable import DevicePassable
from compiler_internal.directives import (
    StaticTensorSpec,
    __mogg_intrinsic_attr,
    StaticTensorSpecInternal,
    get_row_major_tensor_spec_static,
    InputFusion,
    OutputFusion,
    ComputeOutputFusion,
    ElementwiseFusion,
    _NoFusionIn,
    _NoFusionOut,
    _NoComputeFusion,
    _IndexListToTileLayout,
)
from std.gpu.host import get_gpu_target
from std.gpu.host.info import is_cpu
from std.gpu.host.info import is_gpu as _is_gpu
from layout import Coord, LayoutTensor, TileTensor
from layout.tile_layout import Layout as TileLayout, TensorLayout
from register import register_internal
from std.runtime.asyncrt import DeviceContextPtr
from std.runtime.tracing import trace_arg
from tensor import RuntimeTensorSpec

from std.utils import IndexList, StaticTuple
from std.utils._serialize import _serialize

from ._indexing import _dot_prod, _slice_to_tuple
from .io_spec import IO, IOSpec

# ===----------------------------------------------------------------------=== #
# Load / Store Helper primitives
# ===----------------------------------------------------------------------=== #


@parameter
@always_inline
def _gcd_pow2[a: Int, b: Int]() -> Int:
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
@doc_hidden
@register_internal("simd_store_into_managed_tensor_slice")
@always_inline
def simd_store_into_managed_tensor_slice[
    dtype: DType,
    rank: Int,
    simd_width: SIMDSize,
    //,
    static_spec: StaticTensorSpec[dtype, rank, ...],
    element_alignment: Int = 1,
](
    tensor: ManagedTensorSlice[static_spec=static_spec, ...],
    indices: IndexList[rank],
    value: SIMD[dtype, simd_width],
):
    var flat_index = tensor._compute_offset(indices)

    # Store alignment cannot exceed the data type's alignment.
    comptime max_alignment = _gcd_pow2[
        tensor.alignment, element_alignment * align_of[dtype]()
    ]()

    comptime _last_stride_is_static = tensor.static_spec.static_layout._stride_types[
        rank - 1
    ].is_static_value
    comptime _last_stride_value = tensor.static_spec.static_layout._stride_types[
        rank - 1
    ].static_value

    # Stride = 1
    @parameter
    @always_inline
    def store_stride1():
        comptime if dtype == DType.bool:
            var v = value.cast[DType.uint8]()
            tensor._ptr.bitcast[UInt8]().store(flat_index, v)
        else:
            tensor._ptr.store[alignment=max_alignment](flat_index, value)

    # Stride > 1
    @parameter
    @always_inline
    def store_strided(stride: Int):
        comptime if dtype == DType.bool:
            var v = value.cast[DType.uint8]()
            strided_store(
                v,
                tensor._ptr.bitcast[UInt8]() + flat_index,
                stride,
            )
        else:
            return strided_store(value, tensor._ptr + flat_index, stride)

    comptime if not _last_stride_is_static:
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
        comptime if _last_stride_value == 0:
            tensor._ptr.store[alignment=max_alignment](0, value)
        elif _last_stride_value == 1:
            store_stride1()
        else:
            store_strided(_last_stride_value)


@doc_hidden
@register_internal("simd_store_into_tensor_pointer")
@always_inline
def simd_store_into_tensor_pointer[
    dtype: DType,
    rank: Int,
    //,
    static_spec: StaticTensorSpec[dtype, rank, ...],
    simd_width: SIMDSize,
    element_alignment: Int = 1,
](
    ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
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
        simd_width: The SIMD width for the store operation.
        element_alignment: The element alignment for the store.

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
@doc_hidden
@register_internal("simd_load_from_tensor_pointer")
@always_inline
def simd_load_from_tensor_pointer[
    dtype: DType,
    rank: Int,
    //,
    static_spec: StaticTensorSpec[dtype, rank, ...],
    simd_width: Int,
    element_alignment: Int = 1,
](
    ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
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


@doc_hidden
@register_internal("simd_load_from_managed_tensor_slice")
@always_inline
def simd_load_from_managed_tensor_slice[
    dtype: DType,
    rank: Int,
    simd_width: Int,
    //,
    static_spec: StaticTensorSpec[dtype, rank, ...],
    element_alignment: Int = 1,
](
    tensor: ManagedTensorSlice[static_spec=static_spec, ...],
    indices: IndexList[rank],
) -> SIMD[dtype, simd_width]:
    var flat_index = tensor._compute_offset(indices)
    comptime _last_stride_is_static = tensor.static_spec.static_layout._stride_types[
        rank - 1
    ].is_static_value
    comptime _last_stride_value = tensor.static_spec.static_layout._stride_types[
        rank - 1
    ].static_value

    # Load alignment cannot exceed the data type's alignment.
    comptime max_alignment = _gcd_pow2[
        tensor.alignment, element_alignment * align_of[dtype]()
    ]()
    comptime invariant = not tensor.io_spec.mut

    # Stride = 1
    @parameter
    @always_inline
    def load_stride1() -> SIMD[dtype, simd_width]:
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
    def load_strided(stride: Int) -> SIMD[dtype, simd_width]:
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

    comptime if not _last_stride_is_static:
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
        comptime if _last_stride_value == 0:
            return tensor._ptr.load[invariant=invariant](flat_index)
        elif _last_stride_value == 1:
            return load_stride1()
        else:
            return load_strided(_last_stride_value)


# ===----------------------------------------------------------------------=== #
# ManagedTensorSlice class
# ===----------------------------------------------------------------------=== #

comptime OutputTensor = ManagedTensorSlice[io_spec=Output, ...]
comptime InputTensor = ManagedTensorSlice[io_spec=Input, ...]

comptime _MutableInputTensor = ManagedTensorSlice[io_spec=MutableInput, ...]
comptime _FusedOutputTensor = ManagedTensorSlice[io_spec=FusedOutput, ...]
comptime _FusedInputTensor = ManagedTensorSlice[io_spec=FusedInput, ...]

comptime _FusedComputeOutputTensor = ManagedTensorSlice[
    io_spec=_FusedComputeOutput, ...
]

comptime DynamicTensor[dtype: DType, rank: Int] = ManagedTensorSlice[
    io_spec=IOUnknown,
    static_spec=StaticTensorSpec[dtype, rank, ...].get_unknown(),
]


@fieldwise_init
struct ManagedTensorSlice[
    mut: Bool,
    input: IO,
    dtype: DType,
    rank: Int,
    InFusion: InputFusion,
    OutFusion: OutputFusion,
    ComputeFusion: ComputeOutputFusion,
    //,
    io_spec: IOSpec[mut, input],
    *,
    static_spec: StaticTensorSpec[
        dtype, rank, _, InFusion, OutFusion, ComputeFusion
    ],
](DevicePassable, TrivialRegisterPassable, Writable):
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

    def _to_device_type(self, target: MutOpaquePointer[_]):
        target.bitcast[Self.device_type]()[] = self.to_layout_tensor()

    @staticmethod
    def get_type_name() -> String:
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
    # IntTuple aliases for static shape/strides.
    comptime _static_shape_tuple = Self.static_spec.shape_tuple
    comptime _static_strides_tuple = Self.static_spec.strides_tuple

    # Fusion query aliases.
    # _is_unfused and _has_input_fusion are derived purely from IOSpec.
    # _has_output_store_fusion and _has_compute_fusion must inspect the
    # actual type parameters because _bind_to_fused_compute_output (OutputFusion
    # overload) produces the same IO (_FusedComputeOutput) as the
    # ComputeOutputFusion overload but
    # populates OutFusion instead of ComputeFusion.
    comptime _is_unfused: Bool = not Self.input.is_fused()
    comptime _has_input_fusion: Bool = (Self.input == IO.FusedInput)
    comptime _has_output_store_fusion: Bool = not _type_is_eq[
        Self.OutFusion, _NoFusionOut
    ]()
    comptime _has_compute_fusion: Bool = not _type_is_eq[
        Self.ComputeFusion, _NoComputeFusion
    ]()

    var _ptr: UnsafePointer[Scalar[Self.dtype], MutAnyOrigin]
    var _spec: RuntimeTensorSpec[Self.dtype, Self.rank]
    var _runtime_strides: IndexList[Self.rank]
    var in_fusion: Self.InFusion
    var out_fusion: Self.OutFusion
    var compute_fusion: Self.ComputeFusion

    @staticmethod
    @always_inline
    def _sentinel_in_fusion() -> Self.InFusion:
        """Return a sentinel InFusion value, or an uninitialized placeholder
        when the type parameter is a real fusion struct (never reached at
        runtime, but must compile for all instantiations)."""
        comptime if _type_is_eq[Self.InFusion, _NoFusionIn]():
            return rebind[Self.InFusion](_NoFusionIn())
        else:
            var f: Self.InFusion
            __mlir_op.`lit.ownership.mark_initialized`(
                __get_mvalue_as_litref(f)
            )
            return f

    @staticmethod
    @always_inline
    def _sentinel_out_fusion() -> Self.OutFusion:
        """Return a sentinel OutFusion value, or an uninitialized placeholder
        when the type parameter is a real fusion struct."""
        comptime if _type_is_eq[Self.OutFusion, _NoFusionOut]():
            return rebind[Self.OutFusion](_NoFusionOut())
        else:
            var f: Self.OutFusion
            __mlir_op.`lit.ownership.mark_initialized`(
                __get_mvalue_as_litref(f)
            )
            return f

    @staticmethod
    @always_inline
    def _sentinel_compute_fusion() -> Self.ComputeFusion:
        """Return a sentinel ComputeFusion value, or an uninitialized
        placeholder when the type parameter is a real fusion struct."""
        comptime if _type_is_eq[Self.ComputeFusion, _NoComputeFusion]():
            return rebind[Self.ComputeFusion](_NoComputeFusion())
        else:
            var f: Self.ComputeFusion
            __mlir_op.`lit.ownership.mark_initialized`(
                __get_mvalue_as_litref(f)
            )
            return f

    def __init__(
        out self,
        ptr: UnsafePointer[mut=True, Scalar[Self.dtype], _],
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
        def start_fn(slice: Slice) -> Int:
            return slice.start.value()

        @parameter
        @always_inline
        def stop_fn(slice: Slice) -> Int:
            return slice.end.value()

        @parameter
        @always_inline
        def step_fn(slice: Slice) -> Int:
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

        var slicer_strides = adjusted_shape.get_row_major_strides()
        var start_offset = _dot_prod(start, slicer_strides)

        var strides = IndexList[Self.rank]()

        comptime for i in range(Self.rank):
            strides[i] = step[i] * slicer_strides[i]

        self._ptr = ptr + start_offset
        self._spec = slice_spec
        self._runtime_strides = strides
        self.in_fusion = Self._sentinel_in_fusion()
        self.out_fusion = Self._sentinel_out_fusion()
        self.compute_fusion = Self._sentinel_compute_fusion()

    def __init__(
        out self,
        ptr: UnsafePointer[Scalar[Self.dtype], AnyOrigin[mut=True]],
        spec: RuntimeTensorSpec[Self.dtype, Self.rank],
        strides: IndexList[Self.rank],
    ):
        """Initializes a ManagedTensorSlice from a pointer, runtime tensor spec,
        and strides.

        In general, custom operations should not create `ManagedTensorSlice`
        instances, but instead use the ones provided by the MAX inference
        engine.
        """
        self._ptr = ptr
        self._spec = spec
        self._runtime_strides = strides
        self.in_fusion = Self._sentinel_in_fusion()
        self.out_fusion = Self._sentinel_out_fusion()
        self.compute_fusion = Self._sentinel_compute_fusion()

    def __init__(
        out self,
        ptr: UnsafePointer[Scalar[Self.dtype], AnyOrigin[mut=True]],
        shape: IndexList[Self.rank],
    ):
        """Initializes a ManagedTensorSlice from a pointer and shape.

        In general, custom operations should not create `ManagedTensorSlice`
        instances, but instead use the ones provided by the MAX inference
        engine.
        """
        self._ptr = ptr
        self._spec = RuntimeTensorSpec[Self.dtype, Self.rank](shape)
        self._runtime_strides = shape.get_row_major_strides()
        self.in_fusion = Self._sentinel_in_fusion()
        self.out_fusion = Self._sentinel_out_fusion()
        self.compute_fusion = Self._sentinel_compute_fusion()

    def __init__(
        out self,
        ptr: UnsafePointer[Scalar[Self.dtype], AnyOrigin[mut=True]],
        shape: IndexList[Self.rank],
        strides: IndexList[Self.rank],
    ):
        """Initializes a ManagedTensorSlice from a pointer, shape, and strides.

        In general, custom operations should not create `ManagedTensorSlice`
        instances, but instead use the ones provided by the MAX inference
        engine.
        """
        self._ptr = ptr
        self._spec = RuntimeTensorSpec[Self.dtype, Self.rank](shape)
        self._runtime_strides = strides
        self.in_fusion = Self._sentinel_in_fusion()
        self.out_fusion = Self._sentinel_out_fusion()
        self.compute_fusion = Self._sentinel_compute_fusion()

    @always_inline
    def __getitem__(self, indices: IndexList[Self.rank]) -> Scalar[Self.dtype]:
        """Gets the value at the specified indices.

        Args:
          indices: The indices of the value to retrieve.

        Returns:
          The value at the specified indices.
        """
        comptime assert (
            not Self._has_input_fusion
        ), "Direct load on fused tensor is forbidden"
        var offset = _dot_prod(indices, self.strides())
        return self._ptr[offset]

    @always_inline
    def __getitem__(self, *indices: Int) -> Scalar[Self.dtype]:
        """Gets the value at the specified indices.

        Args:
          indices: The indices of the value to retrieve.

        Returns:
          The value at the specified indices.
        """
        comptime assert (
            not Self._has_input_fusion
        ), "Direct load on fused tensor is forbidden"
        assert (
            len(indices) == Self.rank
        ), "mismatch between requested index and rank"
        return self[IndexList[Self.rank](*indices)]

    @always_inline
    def __setitem__(self, *indices: Int, val: Scalar[Self.dtype]):
        """Stores the value at the specified indices.

        Args:
          indices: The indices of the value to store.
          val: The value to store.

        """
        comptime assert (
            not Self._has_output_store_fusion
        ), "Direct store on fused tensor is forbidden"
        assert (
            len(indices) == Self.rank
        ), "mismatch between requested index and rank"
        self[IndexList[Self.rank](*indices)] = val

    @always_inline
    def __setitem__(
        self, indices: IndexList[Self.rank], val: Scalar[Self.dtype]
    ):
        """Stores the value at the specified indices.

        Args:
          indices: The indices of the value to store.
          val: The value to store.

        """
        comptime assert (
            not Self._has_output_store_fusion
        ), "Direct store on fused tensor is forbidden"
        var offset = _dot_prod(indices, self.strides())
        self._ptr[offset] = val

    def spec(self) -> RuntimeTensorSpec[Self.dtype, Self.rank]:
        """Gets the `TensorSpec` of this tensor slice, which provides meta-data
        about the tensor slice.

        Returns:
            The static `TensorSpec` for this tensor slice.
        """
        return self._spec

    @always_inline
    def shape(self) -> IndexList[Self.rank]:
        """Gets the shape of this tensor slice, as an `IndexList`.

        Returns:
            The shape of this tensor slice.
        """
        var result = IndexList[Self.rank]()

        comptime for i in range(Self.rank):
            comptime if Self.static_spec.static_layout._shape_types[
                i
            ].is_static_value:
                result[i] = Self.static_spec.static_layout._shape_types[
                    i
                ].static_value
            else:
                result[i] = self._spec.shape[i]

        return result

    @always_inline
    def dim_size(self, index: Int) -> Int:
        """Gets the size of a given dimension of this tensor slice using a run
        time value.

        Args:
            index: The zero-based index of the dimension.

        Returns:
            The size of the tensor slice in the given dimension.
        """
        return self.shape()[index]

    @always_inline
    def dim_size[index: Int](self) -> Int:
        """Gets the size of a given dimension of this tensor slice using a
        compile time value.

        Parameters:
            index: The zero-based index of the dimension.

        Returns:
            The size of the tensor slice in the given dimension.
        """

        comptime assert 0 <= index < Self.rank, String(
            t"dim_size index of {index} is out of bounds for tensor rank [0,"
            t" {Self.rank}]"
        )

        comptime if not Self.static_spec.static_layout._shape_types[
            index
        ].is_static_value:
            return self._spec.shape[index]
        else:
            return Self.static_spec.static_layout._shape_types[
                index
            ].static_value

    @always_inline
    def strides(self) -> IndexList[Self.rank]:
        """Gets the strides of this tensor slice, as an `IndexList`.

        Returns:
            The strides of this tensor slice.
        """
        var result = IndexList[Self.rank]()

        comptime for i in range(Self.rank):
            comptime if Self.static_spec.static_layout._stride_types[
                i
            ].is_static_value:
                result[i] = Self.static_spec.static_layout._stride_types[
                    i
                ].static_value
            else:
                result[i] = self._runtime_strides[i]

        return result

    @always_inline
    def stride_length(self, index: Int) -> Int:
        """Gets the length of the stride of a given dimension of this tensor
        slice using a run time value.

        Args:
            index: The zero-based index of the dimension.

        Returns:
            The size of the tensor slice in the given dimension.
        """
        return self.strides()[index]

    @always_inline
    def stride_length[index: Int](self) -> Int:
        """Gets the length of the stride of a given dimension of this tensor
        slice using a compile time value.

        Parameters:
            index: The zero-based index of the dimension.

        Returns:
            The size of the tensor slice in the given dimension.
        """

        comptime assert 0 <= index < Self.rank, String(
            t"stride_length index of {index} is out of bounds for tensor rank"
            t" [0, {Self.rank}]"
        )

        comptime if not Self.static_spec.static_layout._stride_types[
            index
        ].is_static_value:
            return self._runtime_strides[index]
        else:
            return Self.static_spec.static_layout._stride_types[
                index
            ].static_value

    @always_inline
    def size(self) -> Int:
        """Computes the tensor slice's number of elements.

        Returns:
            The total number of elements in the tensor slice.
        """
        var product: Int = 1

        comptime for i in range(Self.rank):
            product *= self.dim_size[i]()

        return product

    @always_inline
    def bytecount(self) -> Int:
        """Returns the size of the tensor slice in bytes.

        Returns:
            The total number of bytes in the tensor slice.
        """
        return self.size() * size_of[Self.dtype]()

    @always_inline
    def unsafe_ptr[
        _dtype: DType = Self.dtype
    ](self) -> UnsafePointer[Scalar[_dtype], MutAnyOrigin]:
        """Get the pointer stored in this tensor slice.

        Since this method obtains the pointer stored in this tensor slice, it
        can modify the invariants of this tensor slice and lead to unexpected
        behavior. It should be used with caution.

        Parameters:
            _dtype: The type of the `UnsafePointer` in this tensor slice.

        Returns:
            The `UnsafePointer` which contains the data for this tensor slice.
        """
        return rebind[UnsafePointer[Scalar[_dtype], MutAnyOrigin]](self._ptr)

    @always_inline
    def to_device_buffer(self, ctx: DeviceContext) -> DeviceBuffer[Self.dtype]:
        var size = self.size()
        if size > 0:
            return DeviceBuffer[Self.dtype](
                ctx,
                self.unsafe_ptr(),
                size,
                owning=False,
            )
        else:
            return DeviceBuffer[Self.dtype].empty(ctx)

    @always_inline
    def load[
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
    def _fused_load[
        width: Int,
        # Necessary to make it simpler on the call site.
        _rank: Int,
        element_alignment: Int = 1,
    ](self, index: IndexList[_rank]) -> SIMD[Self.dtype, width]:
        comptime assert _rank == Self.rank
        var ridx = rebind[IndexList[Self.rank]](index)

        comptime if Self._has_input_fusion:
            return self.in_fusion.load[
                Self.dtype, Self.rank, width, element_alignment
            ](ridx)
        else:
            return simd_load_from_managed_tensor_slice[
                simd_width=width, element_alignment=element_alignment
            ](self, ridx)

    @always_inline("nodebug")
    def _lambda_load[
        width: Int,
        _rank: Int,
        element_alignment: Int = 1,
    ](self, index: IndexList[_rank]) -> SIMD[Self.dtype, width]:
        comptime assert _rank == Self.rank
        var ridx = rebind[IndexList[Self.rank]](index)

        comptime assert (
            Self._has_input_fusion
        ), "_lambda_load called on unfused tensor"
        return self.in_fusion.load[
            Self.dtype, Self.rank, width, element_alignment
        ](ridx)

    @always_inline
    def _compute_offset(self, index: IndexList[Self.rank]) -> Int:
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
                comptime if not Self.static_spec.static_layout._stride_types[
                    i
                ].is_static_value:
                    offset = fma(
                        Int32(index[i]), Int32(self._runtime_strides[i]), offset
                    )
                else:
                    offset = fma(
                        Int32(index[i]),
                        Int32(
                            Self.static_spec.static_layout._stride_types[
                                i
                            ].static_value
                        ),
                        offset,
                    )
            return Int(offset)

        var offset = 0

        comptime for i in range(Self.rank):
            comptime if not Self.static_spec.static_layout._stride_types[
                i
            ].is_static_value:
                offset = fma(index[i], self._runtime_strides[i], offset)
            else:
                offset = fma(
                    index[i],
                    Self.static_spec.static_layout._stride_types[
                        i
                    ].static_value,
                    offset,
                )

        return offset

    @always_inline
    def store[
        width: SIMDSize,
        # Necessary to make it simpler on the call site.
        _rank: Int,
        element_alignment: Int = 1,
    ](
        self: ManagedTensorSlice[mut=True, static_spec=Self.static_spec, ...],
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
    def _fused_store[
        width: SIMDSize,
        # Necessary to make it simpler on the call site.
        _rank: Int,
        element_alignment: Int = 1,
    ](
        self: ManagedTensorSlice[mut=True, static_spec=Self.static_spec, ...],
        index: IndexList[_rank],
        val: SIMD[Self.dtype, width],
    ):
        comptime assert _rank == Self.rank
        var ridx = rebind[IndexList[Self.rank]](index)

        comptime if Self._has_output_store_fusion:
            self.out_fusion.store[
                Self.dtype, Self.rank, width, element_alignment
            ](ridx, val)
        else:
            simd_store_into_managed_tensor_slice[
                simd_width=width,
                element_alignment=element_alignment,
            ](self, ridx, val)

    @always_inline("nodebug")
    def _lambda_store[
        width: SIMDSize,
        # Necessary to make it simpler on the call site.
        _rank: Int,
        element_alignment: Int = 1,
    ](
        self: ManagedTensorSlice[
            io_spec=IOSpec[True, Self.input](),
            static_spec=Self.static_spec,
        ],
        index: IndexList[_rank],
        val: SIMD[Self.dtype, width],
    ):
        comptime assert _rank == Self.rank
        var ridx = rebind[IndexList[Self.rank]](index)

        comptime assert (
            Self._has_output_store_fusion
        ), "_lambda_store called on unfused tensor"
        self.out_fusion.store[Self.dtype, Self.rank, width, element_alignment](
            ridx, val
        )

    @always_inline
    def _fused_compute_output_lambda[
        width: SIMDSize,
        # Necessary to make it simpler on the call site.
        _rank: Int,
        element_alignment: Int = 1,
    ](
        self: ManagedTensorSlice[mut=True, static_spec=Self.static_spec, ...],
        index: IndexList[_rank],
        val: SIMD[Self.dtype, width],
    ) -> SIMD[Self.dtype, width]:
        comptime assert _rank == Self.rank
        var ridx = rebind[IndexList[Self.rank]](index)

        comptime if Self._has_compute_fusion:
            return self.compute_fusion.compute[
                Self.dtype, Self.rank, width, element_alignment
            ](ridx, val)
        else:
            return val

    @always_inline
    def with_tile_layout[
        new_layout: TensorLayout,
    ](
        self,
        new_runtime_shape: IndexList[new_layout.rank],
        new_runtime_strides: IndexList[new_layout.rank],
        offset_ptr: Optional[
            UnsafePointer[Scalar[Self.dtype], MutAnyOrigin]
        ] = None,
        out result: ManagedTensorSlice[
            rank=new_layout.rank,
            io_spec=Self.io_spec,
            static_spec=Self.static_spec.with_tile_layout[new_layout](),
        ],
    ):
        return type_of(result)(
            offset_ptr.or_else(self._ptr),
            new_runtime_shape,
            new_runtime_strides,
        )

    @doc_hidden
    @always_inline
    def _bind_to_fused_input[
        F: InputFusion
    ](
        self,
        fusion: F,
        out result: ManagedTensorSlice[
            dtype=Self.dtype,
            rank=Self.rank,
            io_spec=FusedInput,
            static_spec=Self.static_spec.with_input_fusion[F](),
        ],
    ):
        """Bind a trait-based input fusion struct to this tensor.

        The returned MTS dispatches loads through `fusion.load()` instead of
        reading from the underlying data pointer.
        """
        comptime assert (
            Self._is_unfused
        ), "The tensor is already bound to a fusion struct"
        # rebind needed for unfused slots: _is_unfused guarantees the type
        # params equal _NoFusionOut/_NoComputeFusion, but the compiler can't
        # prove it statically.
        return {
            self._ptr,
            self._spec,
            self._runtime_strides,
            fusion,
            rebind[type_of(result).OutFusion](_NoFusionOut()),
            rebind[type_of(result).ComputeFusion](_NoComputeFusion()),
        }

    @doc_hidden
    @always_inline
    def _bind_to_fused_output[
        F: OutputFusion
    ](
        self,
        fusion: F,
        out result: ManagedTensorSlice[
            dtype=Self.dtype,
            rank=Self.rank,
            io_spec=FusedOutput,
            static_spec=Self.static_spec.with_output_fusion[F](),
        ],
    ):
        """Bind a trait-based output fusion struct to this tensor.

        The returned MTS dispatches stores through `fusion.store()` instead of
        writing to the underlying data pointer.
        """
        comptime assert (
            Self._is_unfused
        ), "The tensor is already bound to a fusion struct"
        return {
            self._ptr,
            self._spec,
            self._runtime_strides,
            rebind[type_of(result).InFusion](_NoFusionIn()),
            fusion,
            rebind[type_of(result).ComputeFusion](_NoComputeFusion()),
        }

    @doc_hidden
    @always_inline
    def _bind_to_fused_compute_output[
        F: OutputFusion
    ](
        self,
        fusion: F,
        out result: ManagedTensorSlice[
            dtype=Self.dtype,
            rank=Self.rank,
            io_spec=_FusedComputeOutput,
            static_spec=Self.static_spec.with_output_fusion[F](),
        ],
    ):
        """Bind an OutputFusion struct but with _FusedComputeOutput io_spec.

        Used for the OutputLegacyForCompute case: the kernel expects
        _FusedComputeOutput io_spec but the fusion performs a store.
        """
        comptime assert (
            Self._is_unfused
        ), "The tensor is already bound to a fusion struct"
        return {
            self._ptr,
            self._spec,
            self._runtime_strides,
            rebind[type_of(result).InFusion](_NoFusionIn()),
            fusion,
            rebind[type_of(result).ComputeFusion](_NoComputeFusion()),
        }

    @doc_hidden
    @always_inline
    def _bind_to_fused_compute_output[
        F: ComputeOutputFusion
    ](
        self,
        fusion: F,
        out result: ManagedTensorSlice[
            dtype=Self.dtype,
            rank=Self.rank,
            io_spec=_FusedComputeOutput,
            static_spec=Self.static_spec.with_compute_fusion[F](),
        ],
    ):
        """Bind a trait-based compute-output fusion struct to this tensor.

        The returned MTS dispatches compute-output through
        `fusion.compute()` to transform values before the final store.
        """
        comptime assert (
            Self._is_unfused
        ), "The tensor is already bound to a fusion struct"
        return {
            self._ptr,
            self._spec,
            self._runtime_strides,
            rebind[type_of(result).InFusion](_NoFusionIn()),
            rebind[type_of(result).OutFusion](_NoFusionOut()),
            fusion,
        }

    @always_inline
    def to_layout_tensor(
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
    def to_tile_tensor[
        coord_dtype: DType = DType.int64
    ](
        self,
        out result: TileTensor[
            dtype=Self.dtype,
            origin=MutExternalOrigin,
            LayoutType=TileLayout[
                shape_types=Self.static_spec.static_layout._shape_types,
                stride_types=Self.static_spec.static_layout._stride_types,
            ],
        ],
    ):
        var shape_tuple = Coord[*Self.static_spec.static_layout._shape_types]()
        var stride_tuple = Coord[
            *Self.static_spec.static_layout._stride_types
        ]()
        var shape = self.shape()
        var stride = self.strides()

        comptime for i in range(Self.rank):
            comptime if not shape_tuple.element_types[i].is_static_value:
                shape_tuple[i] = rebind[shape_tuple.element_types[i]](
                    Scalar[shape_tuple.element_types[i].DTYPE](shape[i])
                )

            comptime if not stride_tuple.element_types[i].is_static_value:
                stride_tuple[i] = rebind[stride_tuple.element_types[i]](
                    Scalar[stride_tuple.element_types[i].DTYPE](stride[i])
                )

        return {
            self.unsafe_ptr().unsafe_origin_cast[MutExternalOrigin](),
            TileLayout(shape_tuple, stride_tuple),
        }

    def write_to(self, mut writer: Some[Writer]):
        """
        Formats this buffer to the provided Writer.

        Args:
            writer: The object to write to.
        """
        writer.write("ManagedTensorSlice(")

        @parameter
        def serialize[T: Writable](val: T):
            writer.write(val)

        var shape = List[Int]()
        for i in range(Self.rank):
            shape.append(self.shape()[i])

        # TODO(1937): make this work with all valid strides
        _serialize[serialize_fn=serialize, serialize_end_line=False](
            self._ptr, shape
        )

        writer.write("){")
        writer.write("static_shape = ", self._static_shape_tuple)
        writer.write(", static_strides = ", self._static_strides_tuple)
        writer.write(", dynamic_shape = ", self.shape())
        writer.write(", dynamic_strides = ", self.strides())
        writer.write(", alignment = ", self.alignment)
        writer.write(", address_space = ", self.address_space)
        writer.write("}")

    def write_repr_to(self, mut writer: Some[Writer]):
        """
        Formats this buffer to the provided Writer.

        Args:
            writer: The object to write to.
        """
        self.write_to(writer)


# TODO: Move to oss/modular/mojo/stdlib/stdlib/runtime/tracing.mojo and
# rename to trace_arg
@always_inline
def trace_slice_arg(name: String, buf: ManagedTensorSlice) -> String:
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

comptime InputVariadicTensors = VariadicTensors[io_spec=Input, ...]
comptime OutputVariadicTensors = VariadicTensors[io_spec=Output, ...]

comptime _MutableInputVariadicTensors = VariadicTensors[
    io_spec=MutableInput, ...
]


@fieldwise_init
struct StaticTensorSpecList[
    dtype: DType,
    rank: Int,
    //,
    internals_list: ParameterList[
        type=StaticTensorSpecInternal[dtype, rank], ...
    ],
    shapes_list: ParameterList[type=IndexList[rank], ...],
    strides_list: ParameterList[type=IndexList[rank], ...],
]:
    """A statically indexable list of data that can be assembled into a
    StaticTensorSpecList on demand. This handles the complexities that arise
    with heterogenous specs.
    """

    def __getitem_param__[
        index: Int
    ](
        self,
        out result: StaticTensorSpec[
            Self.dtype,
            Self.rank,
            static_layout=_IndexListToTileLayout[
                Self.shapes_list[index],
                Self.strides_list[index],
            ],
        ],
    ):
        return {Self.internals_list[index]}


@fieldwise_init
struct VariadicTensors[
    mut: Bool,
    input: IO,
    dtype: DType,
    rank: Int,
    //,
    size: Int,
    io_spec: IOSpec[mut, input],
    *,
    static_specs: StaticTensorSpecList[dtype=dtype, rank=rank, ...],
](Sized, TrivialRegisterPassable):
    """A tuple-like container of tensors representing variadic arguments from
    the graph compiler."""

    var _tensors: StaticTuple[DynamicTensor[Self.dtype, Self.rank], Self.size]

    def __init__(
        out self,
        ptrs: StaticTuple[
            UnsafePointer[Scalar[Self.dtype], MutAnyOrigin], Self.size
        ],
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

    def __len__(self) -> Int:
        """Returns the number of variadic arguments in the pack.

        Returns:
            The number of variadic arguments.
        """
        return Self.size

    def __getitem_param__[
        index: Int
    ](
        self,
        out result: ManagedTensorSlice[
            io_spec=Self.io_spec, static_spec=Self.static_specs[index]
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
        return {
            tensor._ptr,
            tensor._spec,
            tensor._runtime_strides,
            _NoFusionIn(),
            _NoFusionOut(),
            _NoComputeFusion(),
        }


# ===----------------------------------------------------------------------=== #
# New VariadicTensors (trait-based fusion, no capturing closures)
# ===----------------------------------------------------------------------=== #


struct _FusionPack[*Ts: TrivialRegisterPassable](TrivialRegisterPassable):
    """TrivialRegisterPassable heterogeneous pack for fusion structs.

    Unlike Tuple, this uses a value-based `!kgen.struct` (not reference-based),
    making it safe to pass across the host-device boundary via GPU closures.
    """

    comptime _mlir_type = __mlir_type[
        `!kgen.struct<`, ~Self.Ts.values, ` isParamPack>`
    ]
    var _mlir_value: Self._mlir_type

    @always_inline("nodebug")
    def __init__(out self, *args: *Self.Ts):
        self._mlir_value = __mlir_op.`kgen.rebind`[_type=Self._mlir_type](
            args.get_loaded_kgen_pack()
        )

    @always_inline("nodebug")
    def __getitem_param__[i: Int](self) -> Self.Ts[i]:
        return __mlir_op.`kgen.struct.extract`[index=i._int_mlir_index()](
            self._mlir_value
        )


struct _FusedInputVariadicTensors[
    dtype: DType,
    rank: Int,
    size: Int,
    //,
    *FusionTypes: InputFusion,
    static_specs: StaticTensorSpecList[dtype=dtype, rank=rank, ...],
](Sized, TrivialRegisterPassable):
    """Variadic input tensors with per-element heterogeneous fusion.

    Tensor data (ptr, shape, strides) is stored in a homogeneous StaticTuple.
    Per-element fusion structs are stored in a _FusionPack, where each
    element conforms to InputFusion. Every element must have a real fusion
    struct — use plain VariadicTensors for unfused variadics.
    """

    var _tensors: StaticTuple[DynamicTensor[Self.dtype, Self.rank], Self.size]
    var _fusions: _FusionPack[*Self.FusionTypes]

    def __init__(
        out self,
        ptrs: StaticTuple[
            UnsafePointer[Scalar[Self.dtype], origin=MutAnyOrigin],
            Self.size,
        ],
        shapes: StaticTuple[IndexList[Self.rank], Self.size],
        fusions: _FusionPack[*Self.FusionTypes],
    ):
        comptime for i in range(Self.size):
            comptime assert not _type_is_eq[
                Self.FusionTypes[i], _NoFusionIn
            ](), (
                "_FusedInputVariadicTensors requires a real fusion struct"
                " for every element; use plain VariadicTensors for unfused"
                " inputs"
            )
        self._tensors = {}
        for i in range(Self.size):
            self._tensors._unsafe_ref(i) = DynamicTensor[Self.dtype, Self.rank](
                ptrs[i], shapes[i]
            )
        self._fusions = fusions

    def __len__(self) -> Int:
        return Self.size

    def __getitem_param__[
        index: Int
    ](
        self,
        out result: ManagedTensorSlice[
            io_spec=FusedInput,
            static_spec=Self.static_specs[index].with_input_fusion[
                Self.FusionTypes[index]
            ](),
        ],
    ):
        """Returns the fused tensor at the given index as a ManagedTensorSlice.

        The returned slice dispatches loads through the element's fusion
        struct, so callers can use `_fused_load` or `_lambda_load` to
        apply the fused computation.

        Parameters:
            index: The index into the variadic tensor arguments.

        Returns:
            The fused tensor at the specified index.
        """
        comptime assert index < Self.size
        var tensor = self._tensors[index]
        return {
            tensor._ptr,
            tensor._spec,
            tensor._runtime_strides,
            self._fusions[index],
            rebind[type_of(result).OutFusion](_NoFusionOut()),
            rebind[type_of(result).ComputeFusion](_NoComputeFusion()),
        }

    def shape[index: Int](self) -> IndexList[Self.rank]:
        """Returns the shape of the tensor at the given index."""
        comptime assert index < Self.size
        return self._tensors[index].shape()

    def get_fusion[index: Int](self) -> Self.FusionTypes[index]:
        """Returns the fusion struct at the given index."""
        return self._fusions[index]


struct _FusedOutputVariadicTensors[
    dtype: DType,
    rank: Int,
    size: Int,
    //,
    *FusionTypes: OutputFusion,
    static_specs: StaticTensorSpecList[dtype=dtype, rank=rank, ...],
](Sized, TrivialRegisterPassable):
    """Variadic output tensors with per-element heterogeneous fusion.

    Tensor data is stored in a homogeneous StaticTuple. Per-element fusion
    structs are stored in a _FusionPack, where each element conforms
    to OutputFusion. Every element must have a real fusion struct — use
    plain VariadicTensors for unfused variadics.
    """

    var _tensors: StaticTuple[DynamicTensor[Self.dtype, Self.rank], Self.size]
    var _fusions: _FusionPack[*Self.FusionTypes]

    def __init__(
        out self,
        ptrs: StaticTuple[
            UnsafePointer[Scalar[Self.dtype], origin=MutAnyOrigin],
            Self.size,
        ],
        shapes: StaticTuple[IndexList[Self.rank], Self.size],
        fusions: _FusionPack[*Self.FusionTypes],
    ):
        comptime for i in range(Self.size):
            comptime assert not _type_is_eq[
                Self.FusionTypes[i], _NoFusionOut
            ](), (
                "_FusedOutputVariadicTensors requires a real fusion struct"
                " for every element; use plain VariadicTensors for unfused"
                " outputs"
            )
        self._tensors = {}
        for i in range(Self.size):
            self._tensors._unsafe_ref(i) = DynamicTensor[Self.dtype, Self.rank](
                ptrs[i], shapes[i]
            )
        self._fusions = fusions

    def __len__(self) -> Int:
        return Self.size

    def __getitem_param__[
        index: Int
    ](
        self,
        out result: ManagedTensorSlice[
            io_spec=FusedOutput,
            static_spec=Self.static_specs[index].with_output_fusion[
                Self.FusionTypes[index]
            ](),
        ],
    ):
        """Returns the fused tensor at the given index as a ManagedTensorSlice.

        The returned slice dispatches stores through the element's fusion
        struct, so callers can use `_lambda_store` to apply the fused
        computation.

        Parameters:
            index: The index into the variadic tensor arguments.

        Returns:
            The fused tensor at the specified index.
        """
        comptime assert index < Self.size
        var tensor = self._tensors[index]
        return {
            tensor._ptr,
            tensor._spec,
            tensor._runtime_strides,
            rebind[type_of(result).InFusion](_NoFusionIn()),
            self._fusions[index],
            rebind[type_of(result).ComputeFusion](_NoComputeFusion()),
        }

    def shape[index: Int](self) -> IndexList[Self.rank]:
        """Returns the shape of the tensor at the given index."""
        comptime assert index < Self.size
        return self._tensors[index].shape()

    def get_fusion[index: Int](self) -> Self.FusionTypes[index]:
        """Returns the fusion struct at the given index."""
        return self._fusions[index]


# ===----------------------------------------------------------------------=== #
# ForEach / view copy primitives
# ===----------------------------------------------------------------------=== #


@doc_hidden
def get_kernel_simd_width[dtype: DType, target: StaticString]() -> Int:
    """Get the simd width used in lambda functions.

    For non-simd arch like GPU, this is the width in terms of number of elements
    used per load/store instruction.
    """

    comptime if _is_gpu[target]():
        # We hardcode simd width to 16B for Nvidia GPUs but >= sm_100
        # arch support 32B load/store to global memory, see KERN-2037.
        comptime if CompilationTarget[get_gpu_target()]._is_arch["sm_100a"]():
            return 32 // size_of[dtype]()

        return simd_width_of[dtype, target=get_gpu_target()]()

    return simd_width_of[dtype]()


@__mogg_intrinsic_attr("mogg.for_each")
@__mogg_intrinsic_attr("mogg.elemwise_for_each")
@no_inline
def foreach[
    dtype: DType,
    rank: Int,
    //,
    func: def[width: Int, element_alignment: Int](
        IndexList[rank]
    ) capturing -> SIMD[dtype, width],
    *,
    target: StaticString = "cpu",
    simd_width: Int = get_kernel_simd_width[dtype, target](),
    _trace_name: StaticString = "mogg.for_each",
](
    tensor: ManagedTensorSlice[mut=True, dtype=dtype, rank=rank, ...],
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

    Args:
        tensor: The output tensor slice which receives the return values from `func`.
        ctx: The call context (forward this from the custom operation).
    """
    assert (
        ctx._handle or is_cpu[target]()
    ), "Expecting non-null device ctx for GPU kernels"

    @parameter
    @always_inline
    def elementwise_fn_wrapper[
        width: Int,
        rank: Int,
        alignment: Int = 1,
    ](index: IndexList[rank]) capturing:
        var val = func[width, alignment](rebind[IndexList[tensor.rank]](index))
        tensor._fused_store[element_alignment=alignment](index, val)

    std.algorithm.functional.elementwise[
        elementwise_fn_wrapper,
        simd_width,
        target=target,
        _trace_description=_trace_name,
    ](tensor.shape(), ctx)


@__mogg_intrinsic_attr("mogg.for_each")
@no_inline
def foreach[
    dtype: DType,
    rank: Int,
    //,
    E: ElementwiseFusion,
    *,
    target: StaticString = "cpu",
    simd_width: Int = get_kernel_simd_width[dtype, target](),
    _trace_name: StaticString = "mogg.for_each",
](
    tensor: ManagedTensorSlice[mut=True, dtype=dtype, rank=rank, ...],
    elem: E,
    ctx: DeviceContextPtr = DeviceContextPtr(),
) raises:
    """Apply a pure elementwise fusion to each element of the tensor slice.

    Parameters:
        dtype: The data type of the elements in the tensor slice.
        rank: The rank of the tensor slice.
        E: The elementwise fusion struct type.
        target: Indicates the type of the target device (e.g. "cpu", "gpu").
        simd_width: The SIMD width for the target.
        _trace_name: Name of the executed operation displayed in the trace.

    Args:
        tensor: The output tensor slice which receives the computed values.
        elem: The elementwise fusion struct.
        ctx: The call context (forward this from the custom operation).
    """

    @parameter
    @always_inline
    def wrapper[
        width: Int, element_alignment: Int
    ](index: IndexList[rank]) capturing -> SIMD[dtype, width]:
        return elem.compute[dtype, rank, width, element_alignment](index)

    foreach[
        func=wrapper,
        target=target,
        simd_width=simd_width,
        _trace_name=_trace_name,
    ](tensor, ctx)


@__mogg_intrinsic_attr("mogg.for_each")
@__mogg_intrinsic_attr("mogg.for_each.out_func")
@no_inline
def foreach[
    dtype: DType,
    rank: Int,
    //,
    func: def[width: Int](IndexList[rank]) capturing -> SIMD[dtype, width],
    out_func: def[width: Int](IndexList[rank]) capturing[_] -> None,
    *,
    target: StaticString = "cpu",
    simd_width: Int = get_kernel_simd_width[dtype, target](),
    _trace_name: StaticString = "mogg.for_each",
](
    tensor: ManagedTensorSlice[dtype=dtype, rank=rank, ...],
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

    Args:
        tensor: The input tensor slice which the consumed values.
        ctx: The call context (forward this from the custom operation).
    """
    assert (
        ctx._handle or is_cpu[target]()
    ), "Expecting non-null device ctx for GPU kernels"

    @parameter
    @always_inline
    def out_func_shim[
        _width: Int, _rank: Int, _alignment: Int = 1
    ](index: IndexList[_rank]) capturing:
        idx = rebind[IndexList[rank]](index)
        out_func[_width](idx)

    std.algorithm.functional.elementwise[
        out_func_shim,
        simd_width,
        target=target,
        _trace_description=_trace_name,
    ](tensor.shape(), ctx)


def foreach[
    dtype: DType,
    rank: Int,
    //,
    func: def[width: Int](IndexList[rank]) capturing -> SIMD[dtype, width],
    *,
    target: StaticString = "cpu",
    simd_width: Int = get_kernel_simd_width[dtype, target](),
    _trace_name: StaticString = "mogg.for_each",
](
    tensor: ManagedTensorSlice[mut=True, dtype=dtype, rank=rank, ...],
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

    Args:
        tensor: The output tensor slice which receives the return values from `func`.
        ctx: The call context (forward this from the custom operation).
    """

    @parameter
    @always_inline
    def func_shim[
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
    ](tensor, ctx)


# TensorCopy intrinsic used by view kernels.
# z is a kernel output, and x a view of the input.
@__mogg_intrinsic_attr("mogg.view_materialize")
@doc_hidden
@no_inline
def view_copy_impl[
    dtype: DType,
    rank: Int,
    InFusion: InputFusion,
    OutFusion: OutputFusion,
    ComputeFusion: ComputeOutputFusion,
    spec: StaticTensorSpec[dtype, rank, _, InFusion, OutFusion, ComputeFusion],
    //,
    *,
    target: StaticString,
    _trace_name: StaticString = "mogg.view_copy_impl",
](
    z: ManagedTensorSlice[mut=True, dtype=dtype, rank=rank, ...],
    x: ManagedTensorSlice[static_spec=spec, ...],
    ctx: DeviceContextPtr,
) raises:
    comptime assert _shape_types_compatible[
        x.static_spec.static_layout._shape_types,
        z.static_spec.static_layout._shape_types,
        rank,
    ](), "static shapes not compatible"
    assert x.shape() == z.shape(), "runtime shapes not compatible"

    @parameter
    @always_inline
    def func[
        width: Int, element_alignment: Int
    ](idx: IndexList[z.rank]) -> SIMD[z.dtype, width]:
        return simd_load_from_managed_tensor_slice[
            simd_width=width, element_alignment=element_alignment
        ](x, idx)

    foreach[
        func,
        target=target,
        _trace_name=_trace_name,
    ](z, ctx)


def _shape_types_compatible[
    x_types: TypeList[Trait=CoordLike, ...],
    y_types: TypeList[Trait=CoordLike, ...],
    rank: Int,
]() -> Bool:
    comptime for i in range(rank):
        comptime if x_types[i].is_static_value and y_types[i].is_static_value:
            comptime if x_types[i].static_value != y_types[i].static_value:
                return False
    return True
