# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from Activations import relu
from Buffer import NDBuffer
from DType import DType
from Functional import elementwise, unroll
from Index import StaticIntTuple
from IO import print, _printf
from List import Dim, DimList
from LLCL import Runtime, OutputChainPtr
from Math import add, div, erf, exp, mul, rsqrt, sqrt, sub, tanh, fma
from Pointer import Pointer, DTypePointer
from Range import range
from SIMD import SIMD
from TargetInfo import simd_width, dtype_simd_width
from TypeUtilities import rebind
from Tracing import Trace, TraceLevel
from String import String

# ===----------------------------------------------------------------------===#
# This file contains the MOGG operation bindings. These should avoid passing
# in complex types as we have to generate the calls using the C++ interface.
#
# They should follow the following guidelines:
# 1. Match the MGP convention of passing in the tensor pointers. (sizes tbd)
# 2. Take attribute parameters at the end and match the MO name exactly. (largely tbd)
# 3. Include new features by importing directly in this file and adding a wrapper which calls it.
# ===----------------------------------------------------------------------===#

# Prevent these functions from being DCE'd by explicitly exporting them.
@export
fn MOGGExport():
    alias _indices = TensorIndicesTypeDef
    alias _out_chain = OutputChainPtrDef
    alias _to_buffer = to_buffer
    alias _add = mogg_add
    alias _div = mogg_div
    alias _erf = mogg_erf
    alias _exp = mogg_exp
    alias _mul = mogg_mul
    alias _rsqrt = mogg_rsqrt
    alias _sqrt = mogg_sqrt
    alias _sub = mogg_sub
    alias _tanh = mogg_tanh
    alias _relu = mogg_relu
    alias _broadcast = broadcast_to_tensor
    alias _simd_load = simd_load
    alias _simd_store = simd_store
    alias _simd_load_1D = simd_load_1D
    alias _simd_load_scalar = simd_load_scalar
    alias _simd_load_splat = simd_load_splat
    alias _simd_load_maybe_splat = simd_load_maybe_splat
    alias _simd_target = get_target_simd
    alias _elementwise = elementwise_wrapper
    alias _print_shape_info = print_buffer_info
    alias _mark_output_chain_ready = mark_output_chain_ready


# Nop functions to expose different types to the compiler.
fn TensorIndicesTypeDef[
    rank: __mlir_type.index
](ty: StaticIntTuple[rank]) -> StaticIntTuple[rank]:
    return ty


fn OutputChainPtrDef(ty: OutputChainPtr) -> OutputChainPtr:
    return ty


@always_inline
fn to_buffer[
    type: DType, rank: __mlir_type.index
](
    data: __mlir_type[`!pop.pointer<scalar<`, type.value, `>>`],
    shape: __mlir_type.`!pop.pointer<index>`,
) -> NDBuffer[rank, DimList[rank].create_unknown(), type]:

    let shape_scalar = __mlir_op.`pop.pointer.bitcast`[
        _type : __mlir_type.`!pop.pointer<!pop.scalar<index>>`
    ](shape)

    var shape_ptr = Pointer(shape)
    var shape_tuple: StaticIntTuple[rank]

    var stride_tuple: StaticIntTuple[rank]
    var stride: Int = 1

    @always_inline
    fn body[idx: Int]():
        # Start from the back so we can accumulate the strides.
        var i = rank - 1 - idx
        shape_tuple[i] = shape_ptr.load(i)
        stride_tuple[i] = stride
        stride *= shape_tuple[i]

    unroll[rank, body]()

    return NDBuffer[rank, DimList[rank].create_unknown(), type](
        data, shape_tuple, DType(type), stride_tuple
    )


fn get_target_simd[type: DType]() -> __mlir_type.index:
    return dtype_simd_width[type]().__as_mlir_index()


@always_inline
fn elementwise_wrapper[
    simd_width: __mlir_type.index,
    type: DType,
    rank: __mlir_type.index,
    func: __mlir_type[
        `!kgen.signature<<`,
        __mlir_type.index,
        `,`,
        __mlir_type.index,
        `>(`,
        StaticIntTuple[
            Int(__mlir_attr[`#kgen.param.index.ref<0, false, 1> : index`])
        ],
        ` borrow) -> !lit.none>`,
    ],
](
    buffer: NDBuffer[rank, DimList[rank].create_unknown(), type],
    out_chain: OutputChainPtr,
):
    @always_inline
    fn mogg_func[simd_width: Int, rank: Int](idx: StaticIntTuple[rank]):
        func[simd_width.__as_mlir_index(), rank.__as_mlir_index()](
            rebind[StaticIntTuple[rank.__as_mlir_index()]](idx)
        )

    alias unroll_factor: Int = 1

    @always_inline
    fn description_fn() -> String:
        let x_separator = String("x")
        let semicolon_separator = String(";")
        let shape_header = String("shape=")
        let unroll_factor_header = String("unroll_factor=")
        let vector_width_header = String("vector_width=")

        let shape_tmp = x_separator.join[rank](buffer.get_shape())
        let shape_str = shape_header.append(shape_tmp)

        let unroll_factor_tmp = String(unroll_factor)
        let unroll_factor_str = unroll_factor_header.append(unroll_factor_tmp)

        let vector_width_tmp = String(simd_width)
        let vector_width_str = vector_width_header.append(vector_width_tmp)

        let res = semicolon_separator.join(
            shape_str, unroll_factor_str, vector_width_str
        )

        x_separator.__del__()
        semicolon_separator.__del__()
        shape_header.__del__()
        unroll_factor_header.__del__()
        vector_width_header.__del__()
        shape_tmp.__del__()
        shape_str.__del__()
        unroll_factor_tmp.__del__()
        unroll_factor_str.__del__()
        vector_width_tmp.__del__()
        vector_width_str.__del__()

        return res

    out_chain.trace_detail[TraceLevel.OP, description_fn]("mogg.element_wise")
    elementwise[rank, simd_width, unroll_factor, mogg_func](
        rebind[StaticIntTuple[rank]](buffer.dynamic_shape),
        out_chain,
    )


# ===----------------------------------------------------------------------===#
# Kernel hooks
# ===----------------------------------------------------------------------===#


@always_inline
fn mogg_add[
    simd_width: __mlir_type.index, type: DType
](x: SIMD[simd_width, type], y: SIMD[simd_width, type]) -> SIMD[
    simd_width, type
]:
    return add(x, y)


@always_inline
fn mogg_div[
    simd_width: __mlir_type.index, type: DType
](x: SIMD[simd_width, type], y: SIMD[simd_width, type]) -> SIMD[
    simd_width, type
]:
    return div(x, y)


@always_inline
fn mogg_erf[
    simd_width: __mlir_type.index, type: DType
](x: SIMD[simd_width, type]) -> SIMD[simd_width, type]:
    return erf(x)


@always_inline
fn mogg_exp[
    simd_width: __mlir_type.index, type: DType
](x: SIMD[simd_width, type]) -> SIMD[simd_width, type]:
    return exp(x)


@always_inline
fn mogg_mul[
    simd_width: __mlir_type.index, type: DType
](x: SIMD[simd_width, type], y: SIMD[simd_width, type]) -> SIMD[
    simd_width, type
]:
    return mul(x, y)


@always_inline
fn mogg_relu[
    simd_width: __mlir_type.index, type: DType
](x: SIMD[simd_width, type]) -> SIMD[simd_width, type]:
    return relu(x)


@always_inline
fn mogg_rsqrt[
    simd_width: __mlir_type.index, type: DType
](x: SIMD[simd_width, type]) -> SIMD[simd_width, type]:
    return rsqrt(x)


@always_inline
fn mogg_sqrt[
    simd_width: __mlir_type.index, type: DType
](x: SIMD[simd_width, type]) -> SIMD[simd_width, type]:
    return sqrt(x)


@always_inline
fn mogg_sub[
    simd_width: __mlir_type.index, type: DType
](x: SIMD[simd_width, type], y: SIMD[simd_width, type]) -> SIMD[
    simd_width, type
]:
    return sub(x, y)


@always_inline
fn mogg_tanh[
    simd_width: __mlir_type.index, type: DType
](x: SIMD[simd_width, type]) -> SIMD[simd_width, type]:
    return tanh(x)


# ===----------------------------------------------------------------------===#
# Simd load/store helper functions
# ===----------------------------------------------------------------------===#


@always_inline
fn _compute_flat_index[
    type: DType, rank: __mlir_type.index, iters: __mlir_type.index
](
    buffer: NDBuffer[rank, DimList[rank].create_unknown(), type],
    index: StaticIntTuple[rank],
) -> Int:
    var flat_index: Int = 0

    @always_inline
    fn body[idx: Int]():
        flat_index = fma(index[idx], buffer.dynamic_stride[idx], flat_index)

    unroll[iters, body]()
    return flat_index


# If we know the tensor is 1D then we can avoid the stride calculation. If
# the stride is 0 then we just splat the value. Hopefully LLVM is able to hoist
# this `if` as it should be a constant.
@always_inline
fn simd_load_1D[
    simd_width: __mlir_type.index, type: DType, rank: __mlir_type.index
](
    buffer: NDBuffer[rank, DimList[rank].create_unknown(), type],
    index: StaticIntTuple[rank],
) -> SIMD[simd_width, type]:
    let stride = buffer.dynamic_stride[rank - 1]
    if stride == 0:
        return simd_load_scalar[simd_width, type, rank](buffer)

    let i = stride * index[rank - 1]
    return buffer.data.simd_load[simd_width](i)


# If we know the tensor is actually a scalar tensor we can avoid all indexing
# calculation.
@always_inline
fn simd_load_scalar[
    simd_width: __mlir_type.index, type: DType, rank: __mlir_type.index
](buffer: NDBuffer[rank, DimList[rank].create_unknown(), type]) -> SIMD[
    simd_width, type
]:
    return buffer.data.load(0)


# Load a tensor which might splat along the last dimension.
@always_inline
fn simd_load_maybe_splat[
    simd_width: __mlir_type.index, type: DType, rank: __mlir_type.index
](
    buffer: NDBuffer[rank, DimList[rank].create_unknown(), type],
    index: StaticIntTuple[rank],
) -> SIMD[simd_width, type]:
    var flat_index = _compute_flat_index[type, rank, rank](buffer, index)

    if buffer.dynamic_stride[rank - 1] == 0:
        return buffer.data.load(flat_index)

    return buffer.data.simd_load[simd_width](flat_index)


# Load a tensor which does a splat along the last dimension.
@always_inline
fn simd_load_splat[
    simd_width: __mlir_type.index, type: DType, rank: __mlir_type.index
](
    buffer: NDBuffer[rank, DimList[rank].create_unknown(), type],
    index: StaticIntTuple[rank],
) -> SIMD[simd_width, type]:
    # Last dimension will be 0 for splats so don't compute last dim.
    var flat_index = _compute_flat_index[
        type, rank, rank - (1).__as_mlir_index()
    ](buffer, index)

    return buffer.data.load(flat_index)


@always_inline
fn simd_load[
    simd_width: __mlir_type.index, type: DType, rank: __mlir_type.index
](
    buffer: NDBuffer[rank, DimList[rank].create_unknown(), type],
    index: StaticIntTuple[rank],
) -> SIMD[simd_width, type]:
    let flat_index = _compute_flat_index[type, rank, rank](buffer, index)
    return buffer.data.simd_load[simd_width](flat_index)


@always_inline
fn simd_store[
    simd_width: __mlir_type.index, type: DType, rank: __mlir_type.index
](
    buffer: NDBuffer[rank, DimList[rank].create_unknown(), type],
    index: StaticIntTuple[rank],
    val: SIMD[simd_width, type],
):
    let flat_index = _compute_flat_index[type, rank, rank](buffer, index)
    buffer.data.simd_store[simd_width](flat_index, val)


# ===----------------------------------------------------------------------===#
# Broadcast
# ===----------------------------------------------------------------------===#


fn broadcast_to_tensor[
    type: DType,
    original_rank: __mlir_type.index,
    target_rank: __mlir_type.index,
    output_rank: __mlir_type.index,
](
    original: NDBuffer[
        original_rank, DimList[original_rank].create_unknown(), type
    ],
    target: NDBuffer[target_rank, DimList[target_rank].create_unknown(), type],
) -> NDBuffer[Int(output_rank), DimList[output_rank].create_unknown(), type]:

    var shape = StaticIntTuple[output_rank]()
    var stride = StaticIntTuple[output_rank]()

    # New dimensions are always broadcast.
    var difference: Int = target_rank - original_rank

    if difference < 0:
        difference = -difference

    for i in range(difference):
        if target_rank >= original_rank:
            shape[i] = target.dim(i)
            stride[i] = 0
        else:
            shape[i] = original.dim(i)
            stride[i] = original.stride(i)

    # Broadcast the remainder as approprate.
    for big_index in range(difference, output_rank, 1):
        # We are traversing as if they are the same size.
        let small_index = big_index - difference

        # Switch the indexes depending on which is bigger.
        var orig_index = small_index
        var target_index = big_index
        if target_rank < original_rank:
            orig_index = big_index
            target_index = small_index

        # If the dims are the same use the stride of the original.
        if original.dim(orig_index) == target.dim(target_index):
            stride[big_index] = original.stride(orig_index)
            shape[big_index] = original.dim(orig_index)
        elif original.dim(orig_index) == 1:
            # If they don't match and original is 1 then we broadcast.
            stride[big_index] = 0
            shape[big_index] = target.dim(target_index)
        else:
            # The target is the one being broadcast here so we don't need to
            # change the strides but we do need to restore the old shape.
            shape[big_index] = original.dim(orig_index)
            stride[big_index] = original.stride(orig_index)

    # Create a view of the original data with the new shape and strides.
    var out = NDBuffer[
        output_rank, DimList[output_rank].create_unknown(), type
    ](
        original.data,
        rebind[StaticIntTuple[output_rank]](shape),
        original.dynamic_dtype,
        rebind[StaticIntTuple[output_rank]](stride),
    )

    return out


# ===----------------------------------------------------------------------===#
# Helpers
# ===----------------------------------------------------------------------===#

# Helper function to mark the output chain as ready in tests.
fn mark_output_chain_ready(out_chain: OutputChainPtr):
    out_chain.mark_ready()


# Helper function to query buffer shapes for tests.
fn print_buffer_info[
    type: DType, rank: __mlir_type.index
](buffer: NDBuffer[rank, DimList[rank].create_unknown(), type],):
    _printf("Rank: ")
    print(rank)
    _printf("Shape: ")
    print(buffer.dynamic_shape)

    _printf("Strides: ")
    print(buffer.dynamic_stride)
