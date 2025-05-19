# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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
from collections import InlineArray, OptionalReg
from collections.string import StaticString
from math import ceil, fma, iota
from random import rand
from sys import alignof, simdwidthof
from sys.info import is_gpu
from sys.intrinsics import strided_load, strided_store

import algorithm
from buffer import DimList, NDBuffer
from buffer.dimlist import _make_partially_static_index_list
from builtin.device_passable import DevicePassable
from compiler_internal.directives import StaticTensorSpec, __mogg_intrinsic_attr
from gpu.host._compile import _get_gpu_target
from gpu.host.info import is_cpu
from gpu.host.info import is_gpu as _is_gpu
from layout import Layout, LayoutTensor, RuntimeLayout
from memory import UnsafePointer
from memory.pointer import _GPUAddressSpace
from register import register_internal
from runtime.asyncrt import DeviceContextPtr
from runtime.tracing import Trace, TraceLevel, trace_arg
from tensor_internal import RuntimeTensorSpec, TensorSpec

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
    constrained[
        a.is_power_of_two() and b.is_power_of_two(),
        "a and b must be powers of 2",
    ]()
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
@no_inline
fn simd_store_into_managed_tensor_slice[
    type: DType,
    rank: Int,
    simd_width: Int,
    static_spec: StaticTensorSpec[type, rank],
    element_alignment: Int = 1,
](
    tensor: ManagedTensorSlice[static_spec=static_spec],
    indices: IndexList[rank],
    value: SIMD[type, simd_width],
):
    var flat_index = tensor._compute_offset(indices)

    # Store alignment cannot exceed the data type's alignment.
    alias max_alignment = _gcd_pow2[
        tensor.alignment, element_alignment * alignof[type]()
    ]()

    alias static_stride = tensor._static_strides.at[rank - 1]()

    # Stride = 1
    @parameter
    @always_inline
    fn store_stride1():
        @parameter
        if type is DType.bool:
            var v = value.cast[DType.uint8]()
            tensor._ptr.bitcast[UInt8]().store(flat_index, v)
        else:
            tensor._ptr.store[alignment=max_alignment](flat_index, value)

    # Stride > 1
    @parameter
    @always_inline
    fn store_strided(stride: Int):
        @parameter
        if type is DType.bool:
            var v = value.cast[DType.uint8]()
            strided_store(
                v,
                tensor._ptr.bitcast[UInt8]().offset(flat_index),
                stride,
            )
        else:
            return strided_store(value, tensor._ptr.offset(flat_index), stride)

    @parameter
    if static_stride.is_dynamic():
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
        @parameter
        if static_stride.get() == 0:
            tensor._ptr.store[alignment=max_alignment](0, value)
        elif static_stride.get() == 1:
            store_stride1()
        else:
            store_strided(static_stride.get())


@doc_private
@register_internal("simd_load_from_managed_tensor_slice")
@no_inline
fn simd_load_from_managed_tensor_slice[
    type: DType,
    rank: Int,
    simd_width: Int,
    static_spec: StaticTensorSpec[type, rank],
](
    tensor: ManagedTensorSlice[static_spec=static_spec],
    indices: IndexList[rank],
) -> SIMD[type, simd_width]:
    var flat_index = tensor._compute_offset(indices)
    alias static_stride = tensor._static_strides.at[rank - 1]()

    # Load alignment cannot exceed the data type's alignment.
    alias max_alignment = _gcd_pow2[tensor.alignment, alignof[type]()]()
    alias invariant = not tensor.io_spec.mut

    # Stride = 1
    @parameter
    @always_inline
    fn load_stride1() -> SIMD[type, simd_width]:
        @parameter
        if type is DType.bool:
            var v = tensor._ptr.bitcast[UInt8]().load[
                width=simd_width,
                invariant=invariant,
            ](flat_index)
            return v.cast[type]()
        else:
            return tensor._ptr.load[
                width=simd_width, alignment=max_alignment, invariant=invariant
            ](flat_index)

    # Stride > 1
    @parameter
    @always_inline
    fn load_strided(stride: Int) -> SIMD[type, simd_width]:
        @parameter
        if type is DType.bool:
            var v = strided_load[simd_width, invariant=invariant](
                tensor._ptr.bitcast[UInt8]().offset(flat_index),
                stride,
            )
            return v.cast[type]()
        else:
            return strided_load[simd_width, invariant=invariant](
                tensor._ptr.offset(flat_index), stride
            )

    @parameter
    if static_stride.is_dynamic():
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
        @parameter
        if static_stride.get() == 0:
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
    type: DType,
    rank: Int, //,
    static_spec: StaticTensorSpec[type, rank],
]() -> __type_of(static_spec):
    return static_spec


@register_internal("rebuild_static_tensor_specs_with_input_lambda")
@no_inline
fn rebuild_static_tensor_specs_with_input_lambda[
    func_type: AnyTrivialRegType, //,
    type: DType,
    rank: Int,
](
    spec: StaticTensorSpec[type, rank],
    in_lambda: func_type,
) -> StaticTensorSpec[type, rank]:
    return StaticTensorSpec[type, rank](
        shape=spec.shape,
        strides=spec.strides,
        alignment=spec.alignment,
        address_space=spec.address_space,
        exclusive=spec.exclusive,
        in_lambda=rebind[spec.in_lambda_t](in_lambda),
        out_lambda=None,
        out_compute_lambda=None,
    )


@register_internal("rebuild_static_tensor_specs_with_output_lambda")
@no_inline
fn rebuild_static_tensor_specs_with_output_lambda[
    func_type: AnyTrivialRegType, //,
    type: DType,
    rank: Int,
](
    spec: StaticTensorSpec[type, rank],
    out_lambda: func_type,
) -> StaticTensorSpec[type, rank]:
    return StaticTensorSpec[type, rank](
        shape=spec.shape,
        strides=spec.strides,
        alignment=spec.alignment,
        address_space=spec.address_space,
        exclusive=spec.exclusive,
        in_lambda=None,
        out_lambda=rebind[spec.out_lambda_t](out_lambda),
        out_compute_lambda=None,
    )


# Helper function used in SliceMOGGDPSFunc to generate the body of the input lambda
@__mogg_intrinsic_attr("mogg.dps_input_fusion_hook")
@register_internal("mogg.dps_input_fusion_hook")
@no_inline
fn _input_fusion_hook_impl[
    mut: Bool, //,
    type: DType,
    rank: Int,
    io_spec: IOSpec[mut],
    static_spec: StaticTensorSpec[type, rank],
](
    tensor: ManagedTensorSlice[io_spec=io_spec, static_spec=static_spec]
) -> __type_of(static_spec):
    @always_inline
    @parameter
    fn _input_lambda[_w: Int](i: IndexList[rank]) -> SIMD[type, _w]:
        # We use these methods to help with fusion passes which manipulates
        # calls. It is helpful to have a registered function.
        return rebind[SIMD[type, _w]](
            simd_load_from_managed_tensor_slice[simd_width=_w](tensor, i)
        )

    return _extract_tensor_spec[
        rebuild_static_tensor_specs_with_input_lambda[type, rank](
            static_spec,
            _input_lambda,
        )
    ]()


# Helper function used in SliceMOGGDPSFunc to generate the body of the output lambda
@__mogg_intrinsic_attr("mogg.dps_output_fusion_hook")
@register_internal("mogg.dps_output_fusion_hook")
@no_inline
fn _output_fusion_hook_impl[
    mut: Bool, //,
    type: DType,
    rank: Int,
    io_spec: IOSpec[mut],
    static_spec: StaticTensorSpec[type, rank],
](
    tensor: ManagedTensorSlice[io_spec=io_spec, static_spec=static_spec]
) -> __type_of(static_spec):
    @always_inline
    @parameter
    fn _output_lambda[
        _w: Int, _elem_align: Int = 1
    ](i: IndexList[rank], v: SIMD[type, _w]):
        # We use these methods to help with fusion passes which manipulates
        # calls. It is helpful to have a registered function.
        simd_store_into_managed_tensor_slice[
            simd_width=_w,
            element_alignment=_elem_align,
        ](tensor, i, rebind[SIMD[type, _w]](v))

    return _extract_tensor_spec[
        rebuild_static_tensor_specs_with_output_lambda[type, rank](
            static_spec,
            _output_lambda,
        )
    ]()


# ===----------------------------------------------------------------------=== #
# Mixed precision output fusion hook used by experimental codegen path.
# ===----------------------------------------------------------------------=== #


@no_inline
fn rebuild_mix_precision_static_tensor_specs_with_output_lambda[
    func_type: AnyTrivialRegType, //,
    dst_type: DType,
    src_type: DType,
    rank: Int,
](
    spec: StaticTensorSpec[dst_type, rank],
    out_lambda: func_type,
    out result: StaticTensorSpec[src_type, rank],
):
    return StaticTensorSpec[src_type, rank](
        shape=spec.shape,
        strides=spec.strides,
        alignment=spec.alignment,
        address_space=spec.address_space,
        exclusive=spec.exclusive,
        in_lambda=None,
        out_lambda=rebind[result.out_lambda_t](out_lambda),
        out_compute_lambda=None,
    )


@__mogg_intrinsic_attr("mogg.dps_mixed_precision_output_fusion_hook")
@register_internal("mogg.dps_mixed_precision_output_fusion_hook")
@no_inline
fn _mixed_precision_output_fusion_hook_impl[
    mut: Bool, //,
    dst_type: DType,  # The DType after casting.
    src_type: DType,  # The DType before casting.
    rank: Int,
    io_spec: IOSpec[mut],
    static_spec: StaticTensorSpec[dst_type, rank],
](
    tensor: ManagedTensorSlice[io_spec=io_spec, static_spec=static_spec]
) -> StaticTensorSpec[src_type, rank]:
    @always_inline
    @parameter
    fn _output_lambda[
        _w: Int, _elem_align: Int = 1
    ](i: IndexList[rank], v: SIMD[src_type, _w]):
        # .... compiler-generated-code insert here!
        simd_store_into_managed_tensor_slice[
            simd_width=_w,
            element_alignment=_elem_align,
        ](tensor, i, rebind[SIMD[dst_type, _w]](v))

    return _extract_tensor_spec[
        rebuild_mix_precision_static_tensor_specs_with_output_lambda[
            dst_type, src_type, rank
        ](
            static_spec,
            _output_lambda,
        )
    ]()


@register_internal(
    "rebuild_mix_precision_static_tensor_specs_with_input_lambda"
)
@no_inline
fn rebuild_mix_precision_static_tensor_specs_with_input_lambda[
    func_type: AnyTrivialRegType, //,
    src_type: DType,
    dst_type: DType,
    rank: Int,
](
    spec: StaticTensorSpec[src_type, rank],
    in_lambda: func_type,
    out result: StaticTensorSpec[dst_type, rank],
):
    return StaticTensorSpec[dst_type, rank](
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
    mut: Bool, //,
    dst_type: DType,  # The DType after casting.
    src_type: DType,  # The DType before casting.
    rank: Int,
    io_spec: IOSpec[mut],
    static_spec: StaticTensorSpec[src_type, rank],
](
    tensor: ManagedTensorSlice[io_spec=io_spec, static_spec=static_spec]
) -> StaticTensorSpec[dst_type, rank]:
    @always_inline
    @parameter
    fn _input_lambda[_w: Int](i: IndexList[rank]) -> SIMD[dst_type, _w]:
        # We use these methods to help with fusion passes which manipulates
        # calls. It is helpful to have a registered function.
        var v = rebind[SIMD[src_type, _w]](
            simd_load_from_managed_tensor_slice[simd_width=_w](tensor, i)
        )
        # .... compiler-generated-code here to bridge between src and dst_type
        return rebind[SIMD[dst_type, _w]](v)

    return _extract_tensor_spec[
        rebuild_mix_precision_static_tensor_specs_with_input_lambda[
            src_type, dst_type, rank
        ](
            static_spec,
            _input_lambda,
        )
    ]()


# ===----------------------------------------------------------------------=== #
# ManagedTensorSlice class
# ===----------------------------------------------------------------------=== #

alias OutputTensor = ManagedTensorSlice[io_spec=Output]
alias InputTensor = ManagedTensorSlice[io_spec=Input]

alias _MutableInputTensor = ManagedTensorSlice[io_spec=MutableInput]
alias _FusedOutputTensor = ManagedTensorSlice[io_spec=FusedOutput]
alias _FusedInputTensor = ManagedTensorSlice[io_spec=FusedInput]

alias _FusedComputeOutputTensor = ManagedTensorSlice[
    io_spec=_FusedComputeOutput
]


struct DynamicTensor[
    type: DType,
    rank: Int,
]:
    alias Type = ManagedTensorSlice[
        io_spec=IOUnknown,
        static_spec = StaticTensorSpec[type, rank].create_unknown(),
    ]


@value
@register_passable("trivial")
struct ManagedTensorSlice[
    mut: Bool,
    input: IO,
    type: DType,
    rank: Int, //,
    io_spec: IOSpec[mut, input],
    *,
    static_spec: StaticTensorSpec[type, rank],
](Copyable, DevicePassable, Movable, Stringable, Writable):
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
    alias device_type: AnyTrivialRegType = LayoutTensor[
        type, static_spec.to_layout(), MutableAnyOrigin
    ]

    fn _to_device_type(self, target: UnsafePointer[NoneType]):
        target.bitcast[Self.device_type]()[] = self.to_layout_tensor()

    @staticmethod
    fn get_type_name() -> String:
        return (
            "ManagedTensorSlice[mut = "
            + String(mut)
            + ", type = "
            + String(type)
            + ", rank = "
            + String(rank)
            + ", static_spec (as Layout) = "
            + String(static_spec.to_layout())
            + "]"
        )

    @staticmethod
    fn get_device_type_name() -> String:
        return (
            "LayoutTensor[mut = "
            + String(Self.device_type.mut)
            + ", dtype = "
            + String(Self.device_type.dtype)
            + ", layout = "
            + String(Self.device_type.layout)
            + ", address_space = "
            + String(Self.device_type.address_space)
            + "]"
        )

    alias address_space = static_spec.address_space
    alias alignment = static_spec.alignment
    alias exclusive = static_spec.exclusive
    alias _static_shape = static_spec.shape
    alias _static_strides = static_spec.strides

    alias _in_lambda = static_spec.in_lambda
    alias _out_lambda = static_spec.out_lambda

    var _ptr: UnsafePointer[Scalar[type]]
    var _spec: RuntimeTensorSpec[type, rank]
    var _runtime_strides: IndexList[rank]

    fn __init__(
        out self,
        ptr: UnsafePointer[Scalar[type]],
        slices: InlineArray[Slice, rank],
        slicer_spec: RuntimeTensorSpec[type, rank],
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

        var adjusted_shape = IndexList[rank]()
        for i in range(rank):
            adjusted_shape[i] = Int(ceil((stop[i] - start[i]) / step[i]))
        var slice_spec = RuntimeTensorSpec[type](adjusted_shape)

        var slicer_strides = _row_major_strides(adjusted_shape)
        var start_offset = _dot_prod(start, slicer_strides)

        var strides = IndexList[rank]()

        @parameter
        for i in range(rank):
            strides[i] = step[i] * slicer_strides[i]

        self = Self(ptr.offset(start_offset), slice_spec, strides)

    fn __init__(
        out self,
        ptr: UnsafePointer[Scalar[type]],
        shape: IndexList[rank],
    ):
        """Initializes a ManagedTensorSlice from a pointer and shape.

        In general, custom operations should not create `ManagedTensorSlice`
        instances, but instead use the ones provided by the MAX inference
        engine.
        """
        self._ptr = ptr
        self._spec = RuntimeTensorSpec[type, rank](shape)
        self._runtime_strides = _row_major_strides(shape)

    fn __init__(
        out self,
        ptr: UnsafePointer[Scalar[type]],
        shape: IndexList[rank],
        strides: IndexList[rank],
    ):
        """Initializes a ManagedTensorSlice from a pointer, shape, and strides.

        In general, custom operations should not create `ManagedTensorSlice`
        instances, but instead use the ones provided by the MAX inference
        engine.
        """
        self = Self(
            ptr,
            RuntimeTensorSpec[type, rank](shape),
            strides,
        )

    @doc_private
    @implicit
    fn __init__(out self, ndbuffer: NDBuffer[type, rank]):
        """Initializes a ManagedTensorSlice from an NDBuffer.

        Note that forwarding of static shape, strides, and lambdas won't work.
        """
        self = Self(ndbuffer.data, ndbuffer.get_shape())

    @always_inline
    fn __getitem__(self, indices: IndexList[rank]) -> Scalar[type]:
        """Gets the value at the specified indices.

        Args:
          indices: The indices of the value to retrieve.

        Returns:
          The value at the specified indices.
        """
        constrained[
            not static_spec.in_lambda,
            "Direct load on fused tensor is forbidden",
        ]()
        var offset = _dot_prod(indices, self.strides())
        return self._ptr[offset]

    @always_inline
    fn __getitem__(self, *indices: Int) -> Scalar[type]:
        """Gets the value at the specified indices.

        Args:
          indices: The indices of the value to retrieve.

        Returns:
          The value at the specified indices.
        """
        constrained[
            not static_spec.in_lambda,
            "Direct load on fused tensor is forbidden",
        ]()
        debug_assert(
            len(indices) == rank, "mismatch between requested index and rank"
        )
        return self[indices]

    @always_inline
    fn __setitem__(self, *indices: Int, val: Scalar[type]):
        """Stores the value at the specified indices.

        Args:
          indices: The indices of the value to store.
          val: The value to store.

        """
        constrained[
            not static_spec.out_lambda,
            "Direct store on fused tensor is forbidden",
        ]()
        debug_assert(
            len(indices) == rank, "mismatch between requested index and rank"
        )
        self[indices] = val

    @always_inline
    fn __setitem__(self, indices: IndexList[rank], val: Scalar[type]):
        """Stores the value at the specified indices.

        Args:
          indices: The indices of the value to store.
          val: The value to store.

        """
        constrained[
            not static_spec.out_lambda,
            "Direct store on fused tensor is forbidden",
        ]()
        var offset = _dot_prod(indices, self.strides())
        self._ptr[offset] = val

    fn spec(self) -> RuntimeTensorSpec[type, rank]:
        """Gets the `TensorSpec` of this tensor slice, which provides meta-data
        about the tensor slice.

        Returns:
            The static `TensorSpec` for this tensor slice.
        """
        return self._spec

    @always_inline
    fn shape(self) -> IndexList[rank]:
        """Gets the shape of this tensor slice, as an `IndexList`.

        Returns:
            The shape of this tensor slice.
        """
        return _make_partially_static_index_list[rank, Self._static_shape](
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

        @parameter
        if Self._static_shape.at[index]().is_dynamic():
            return self._spec.shape[index]
        else:
            return Self._static_shape.get[index]()

    @always_inline
    fn strides(self) -> IndexList[rank]:
        """Gets the strides of this tensor slice, as an `IndexList`.

        Returns:
            The strides of this tensor slice.
        """
        return _make_partially_static_index_list[rank, Self._static_strides](
            self._runtime_strides
        )

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

        @parameter
        if Self._static_strides.at[index]().is_dynamic():
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

        @parameter
        for i in range(rank):
            product *= self.dim_size[i]()

        return product

    @always_inline
    fn unsafe_ptr[__type: DType = type](self) -> UnsafePointer[Scalar[__type]]:
        """Get the pointer stored in this tensor slice.

        Since this method obtains the pointer stored in this tensor slice, it
        can modify the invariants of this tensor slice and lead to unexpected
        behavior. It should be used with caution.

        Parameters:
            __type: The type of the `UnsafePointer` in this tensor slice.

        Returns:
            The `UnsafePointer` which contains the data for this tensor slice.
        """
        return rebind[UnsafePointer[Scalar[__type]]](self._ptr)

    @always_inline
    fn load[
        width: Int,
        # Necessary to make it simpler on the call site.
        _rank: Int,
    ](self, index: IndexList[_rank]) -> SIMD[type, width]:
        """Gets data from this tensor slice as a `SIMD`.

        Parameters:
            width: The width of the `SIMD` value. This must be large enough to contain the data from this tensor slice.
            _rank: The rank of the tensor slice.

        Args:
            index: An `IndexList` of size `_rank` to indicate the dimension of the tensor slice to obtain data from.

        Returns:
            Data from this tensor slice at dimension `index`.
        """
        constrained[
            input == IO.Input or input == IO.Unknown,
            "loading not supported for output tensors",
        ]()

        constrained[_rank == rank]()
        var ridx = rebind[IndexList[rank]](index)
        return simd_load_from_managed_tensor_slice[simd_width=width](self, ridx)

    @__mogg_intrinsic_attr("mogg.tensor_fused_load")
    @always_inline
    fn _fused_load[
        width: Int,
        # Necessary to make it simpler on the call site.
        _rank: Int,
    ](self, index: IndexList[_rank]) capturing -> SIMD[type, width]:
        constrained[_rank == rank]()
        var ridx = rebind[IndexList[rank]](index)

        alias in_lambda = static_spec.in_lambda
        alias alignment = static_spec.alignment
        alias address_space = static_spec.address_space
        alias strides = static_spec.strides

        @parameter
        if in_lambda:
            alias in_fn = in_lambda.value()
            return in_fn[width](ridx)
        else:
            return simd_load_from_managed_tensor_slice[simd_width=width](
                self, ridx
            )

    @always_inline("nodebug")
    fn _lambda_load[
        width: Int,
        # Necessary to make it simpler on the call site.
        _rank: Int,
    ](self, index: IndexList[_rank]) -> SIMD[type, width]:
        constrained[_rank == rank]()
        var ridx = rebind[IndexList[rank]](index)
        alias in_lambda = static_spec.in_lambda
        constrained[Bool(in_lambda)]()
        alias in_fn = in_lambda.value()
        return in_fn[width](ridx)

    @always_inline
    fn _compute_offset(self, index: IndexList[rank]) -> Int:
        @parameter
        if rank == 0:
            return 0

        # Special case for NVidia GPU on shared memory.
        # We can do the offset computation in int32 instead.
        @parameter
        if is_gpu() and Self.address_space in (
            _GPUAddressSpace.SHARED,
            _GPUAddressSpace.LOCAL,
            _GPUAddressSpace.CONSTANT,
        ):
            var offset: Int32 = 0

            @parameter
            for i in range(rank):

                @parameter
                if Self._static_strides.at[i]().is_dynamic():
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

        @parameter
        for i in range(rank):

            @parameter
            if Self._static_strides.at[i]().is_dynamic():
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
            static_spec=static_spec,
        ],
        index: IndexList[_rank],
        val: SIMD[type, width],
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
        constrained[_rank == rank]()
        var ridx = rebind[IndexList[rank]](index)

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
        self: ManagedTensorSlice[mut=True, static_spec=static_spec],
        index: IndexList[_rank],
        val: SIMD[type, width],
    ) capturing:
        constrained[_rank == rank]()
        var ridx = rebind[IndexList[rank]](index)

        alias out_lambda = static_spec.out_lambda
        alias alignment = static_spec.alignment
        alias address_space = static_spec.address_space
        alias strides = static_spec.strides

        @parameter
        if out_lambda:
            alias out_fn = out_lambda.value()
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
            io_spec = IOSpec[True, input](),
            static_spec=static_spec,
        ],
        index: IndexList[_rank],
        val: SIMD[type, width],
    ):
        constrained[_rank == rank]()
        var ridx = rebind[IndexList[rank]](index)
        alias out_lambda = static_spec.out_lambda
        constrained[Bool(out_lambda)]()
        alias out_fn = out_lambda.value()
        out_fn[width, element_alignment](ridx, val)

    @always_inline
    fn _fused_compute_output_lambda[
        width: Int,
        # Necessary to make it simpler on the call site.
        _rank: Int,
    ](
        self: ManagedTensorSlice[mut=True, static_spec=static_spec],
        index: IndexList[_rank],
        val: SIMD[type, width],
    ) capturing -> SIMD[type, width]:
        constrained[_rank == rank]()
        var ridx = rebind[IndexList[rank]](index)

        alias out_compute_lambda = static_spec.out_compute_lambda

        @parameter
        if out_compute_lambda:
            alias out_fn = out_compute_lambda.value()
            return out_fn[width](ridx, val)
        else:
            return val

    @always_inline
    fn with_layout[
        new_rank: Int, //,
        new_static_shape: DimList,
        new_static_strides: DimList,
    ](
        self,
        new_runtime_shape: IndexList[new_rank],
        new_runtime_strides: IndexList[new_rank],
        offset_ptr: OptionalReg[UnsafePointer[Scalar[type]]] = None,
        out result: ManagedTensorSlice[
            rank=new_rank,
            io_spec=io_spec,
            static_spec = static_spec.with_layout[new_rank](
                new_static_shape, new_static_strides
            ),
        ],
    ):
        constrained[
            len(new_static_shape) == new_rank, "static shape has incorrect rank"
        ]()
        constrained[
            len(new_static_strides) == new_rank,
            "static strides has incorrect rank",
        ]()
        debug_assert(
            _is_consistent[new_static_shape](new_runtime_shape)
            and _is_consistent[new_static_strides](new_runtime_strides)
        )

        return __type_of(result)(
            offset_ptr.or_else(self._ptr),
            new_runtime_shape,
            new_runtime_strides,
        )

    @always_inline
    fn to_layout_tensor(
        self,
        out result: LayoutTensor[
            type, static_spec.to_layout(), MutableAnyOrigin
        ],
    ):
        alias layout = static_spec.to_layout()
        return LayoutTensor[type, layout](
            self.unsafe_ptr(),
            __type_of(result.runtime_layout)(
                self.shape().cast[result.layout_int_type](),
                self.strides().cast[result.linear_idx_type](),
            ),
        )

    fn write_to[W: Writer](self, mut writer: W):
        """
        Formats this buffer to the provided Writer.

        Parameters:
            W: A type conforming to the Writable trait.

        Args:
            writer: The object to write to.
        """
        writer.write("ManagedTensorSlice(")

        @parameter
        fn serialize[T: Writable](val: T):
            writer.write(val)

        var shape = List[Int, hint_trivial_type=True]()
        for i in range(rank):
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
    @parameter
    if len(static_info) != runtime_info.size:
        return False

    @parameter
    for i in range(runtime_info.size):

        @parameter
        if not static_info.has_value[i]():
            continue

        if static_info.at[i]() != runtime_info[i]:
            return False

    return True


# TODO: Move to open-source/max/mojo/stdlib/stdlib/runtime/tracing.mojo and
# rename to trace_arg
@always_inline
fn trace_slice_arg(name: String, buf: ManagedTensorSlice) -> String:
    """Helper to stringify the type and shape of a kernel argument for tracing.

    Args:
        name: The name of the argument.
        buf: The NDBuffer to trace.

    Returns:
        A string representation of the buffer with its shape and data type.
    """
    return trace_arg(name, buf._runtime_strides, buf.type)


# ===----------------------------------------------------------------------=== #
# VariadicTensors
# ===----------------------------------------------------------------------=== #

alias InputVariadicTensors = VariadicTensors[io_spec=Input]
alias OutputVariadicTensors = VariadicTensors[io_spec=Output]

alias _MutableInputVariadicTensors = VariadicTensors[io_spec=MutableInput]
alias _FusedInputVariadicTensors = VariadicTensors[io_spec=FusedInput]
alias _FusedOutputVariadicTensors = VariadicTensors[io_spec=FusedOutput]


@value
@register_passable("trivial")
struct VariadicTensors[
    mut: Bool,
    input: IO, //,
    type: DType,
    rank: Int,
    size: Int,
    io_spec: IOSpec[mut, input],
    *,
    static_specs: StaticTuple[StaticTensorSpec[type, rank], size],
](Sized):
    """A tuple-like container of tensors representing variadic arguments from
    the graph compiler."""

    var _tensors: StaticTuple[DynamicTensor[type, rank].Type, size]

    fn __len__(self) -> Int:
        """Returns the number of variadic arguments in the pack.

        Returns:
            The number of variadic arguments.
        """
        return size

    fn __getitem__[
        index: Int
    ](
        self,
        out result: ManagedTensorSlice[
            io_spec=io_spec, static_spec = static_specs[index]
        ],
    ):
        """Returns the tensor at the given position in the variadic argument
        argument pack.

        Parameters:
            index: The index into the variadic tensor arguments.

        Returns:
            The tensor at the specified index.
        """
        constrained[index < size]()
        var tensor = self._tensors[index]
        return __type_of(result)(
            tensor._ptr, tensor._spec, tensor._runtime_strides
        )


# ===----------------------------------------------------------------------=== #
# ForEach / view copy primitives
# ===----------------------------------------------------------------------=== #


@doc_private
fn get_kernel_simd_width[type: DType, target: StaticString]() -> Int:
    @parameter
    if _is_gpu[target]():
        return simdwidthof[type, target = _get_gpu_target()]()

    return simdwidthof[type]()


@__mogg_intrinsic_attr("mogg.for_each")
@__mogg_intrinsic_attr("mogg.elemwise_for_each")
@no_inline
fn foreach[
    type: DType,
    rank: Int, //,
    func: fn[width: Int] (IndexList[rank]) capturing -> SIMD[type, width],
    *,
    target: StaticString = "cpu",
    simd_width: Int = get_kernel_simd_width[type, target](),
    _synchronous: Bool = False,
    _trace_name: StaticString = "mogg.for_each",
](
    tensor: ManagedTensorSlice[mut=True, type=type, rank=rank],
    ctx: DeviceContextPtr = DeviceContextPtr(),
) raises:
    """Apply the function `func` to each element of the tensor slice.

    Parameters:
        type: The data type of the elements in the tensor slice.
        rank: The rank of the tensor slice.
        func: The function to apply to each element of the tensor slice.
        target: Indicates the type of the target device (e.g. "cpu", "gpu").
        simd_width: The SIMD width for the target (usually leave this as its default value).
        _synchronous: True to run the custom op synchronously in the runtime (defaults to False).
        _trace_name: Name of the executed operation displayed in the trace_description.

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
        width: Int, rank: Int
    ](index: IndexList[rank]) capturing:
        var val = func[width](rebind[IndexList[tensor.rank]](index))
        tensor._fused_store(index, val)

    algorithm.functional.elementwise[
        elementwise_fn_wrapper,
        simd_width,
        use_blocking_impl=_synchronous,
        target=target,
        _trace_description=_trace_name,
    ](tensor.shape(), ctx)


@__mogg_intrinsic_attr("mogg.for_each")
@__mogg_intrinsic_attr("mogg.for_each.out_func")
@no_inline
fn foreach[
    type: DType,
    rank: Int, //,
    func: fn[width: Int] (IndexList[rank]) capturing -> SIMD[type, width],
    out_func: fn[width: Int] (IndexList[rank]) capturing [_] -> None,
    *,
    target: StaticString = "cpu",
    simd_width: Int = get_kernel_simd_width[type, target](),
    _synchronous: Bool = False,
    _trace_name: StaticString = "mogg.for_each",
](
    tensor: ManagedTensorSlice[type=type, rank=rank],
    ctx: DeviceContextPtr = DeviceContextPtr(),
) raises:
    """Apply the function `func` to each element of the tensor slice.

    Parameters:
        type: The data type of the elements in the tensor slice.
        rank: The rank of the tensor slice.
        func: The function to apply to each element of the tensor slice.
        out_func: The function to apply on each output element.
        target: Indicates the type of the target device (e.g. "cpu", "gpu").
        simd_width: The SIMD width for the target (usually leave this as its default value).
        _synchronous: True to run the custom op synchronously in the runtime (defaults to False).
        _trace_name: Name of the executed operation displayed in the trace_description.

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
        _width: Int, _rank: Int
    ](index: IndexList[_rank]) capturing:
        idx = rebind[IndexList[rank]](index)
        out_func[_width](idx)

    algorithm.functional.elementwise[
        out_func_shim,
        simd_width,
        use_blocking_impl=_synchronous,
        target=target,
        _trace_description=_trace_name,
    ](tensor.shape(), ctx)


# TensorCopy intrinsic used by view kernels.
# z is a kernel output, and x a view of the input.
@__mogg_intrinsic_attr("mogg.view_materialize")
@doc_private
@no_inline
fn view_copy_impl[
    type: DType,
    rank: Int,
    spec: StaticTensorSpec[type, rank], //,
    *,
    target: StaticString,
    _synchronous: Bool,
    _trace_name: StaticString = "mogg.view_copy_impl",
](
    z: ManagedTensorSlice[mut=True, type=type, rank=rank],
    x: ManagedTensorSlice[static_spec=spec],
    ctx: DeviceContextPtr,
) raises:
    constrained[
        _compatible_with[x._static_shape, z._static_shape](),
        "static shapes not compatible",
    ]()
    debug_assert(x.shape() == z.shape(), "runtime shapes not compatible")

    @parameter
    @always_inline
    fn func[width: Int](idx: IndexList[z.rank]) -> SIMD[z.type, width]:
        return simd_load_from_managed_tensor_slice[simd_width=width](x, idx)

    foreach[
        func,
        target=target,
        _synchronous=_synchronous,
        _trace_name=_trace_name,
    ](z, ctx)


fn _compatible_with[x: DimList, y: DimList]() -> Bool:
    @parameter
    if len(x) != len(y):
        return False

    @parameter
    for i in range(len(x)):
        if x.has_value[i]() and y.has_value[i]() and x.at[i]() != y.at[i]():
            return False

    return True
