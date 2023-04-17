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

# Prevent these functions from being DCE'd by explicitly exporting them.
@export
fn MOGGExport():
    alias _indices = TensorIndicesTypeDef
    alias _out_chain = OutputChainPtrDef
    alias _simd_typedef = SimdTypeDef
    alias _index_typedef = IndexTypeDef
    alias _dtype_f32 = DTypeF32TypeDef
    alias _dtype_si32 = DTypeSI32TypeDef
    alias _to_buffer = to_buffer
    alias _add = add
    alias _div = div
    alias _erf = erf
    alias _exp = exp
    alias _load_scalar = load_scalar
    alias _mul = mul
    alias _rsqrt = rsqrt
    alias _sqrt = sqrt
    alias _sub = sub
    alias _tanh = tanh
    alias _relu = relu
    alias _broadcast = broadcast_to_tensor
    alias _simd_load = simd_load
    alias _simd_store = simd_store
    alias _simd_load_1D = simd_load_1D
    alias _simd_load_splat = simd_load_splat
    alias _simd_load_maybe_splat = simd_load_maybe_splat
    alias _simd_target = dtype_simd_width
    alias _splat = splat
    alias _elementwise = elementwise_wrapper
    alias _print_shape_info = print_buffer_info
    alias _mark_output_chain_ready = mark_output_chain_ready


# ===----------------------------------------------------------------------===#
# Nop functions to expose different types to the compiler.
# ===----------------------------------------------------------------------===#


fn DTypeF32TypeDef(ty: DType.type) -> DType.type:
    return DType.f32.value


fn DTypeSI32TypeDef(ty: DType.type) -> DType.type:
    return DType.si32.value


fn IndexTypeDef(ty: Int) -> Int:
    return ty


fn OutputChainPtrDef(ty: OutputChainPtr) -> OutputChainPtr:
    return ty


fn SimdTypeDef[
    type: DType, simd_width: Int
](ty: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
    return ty


fn TensorIndicesTypeDef[
    rank: Int
](ty: StaticIntTuple[rank]) -> StaticIntTuple[rank]:
    return ty


# ===----------------------------------------------------------------------===#
# Basic generated kernel building blocks
# ===----------------------------------------------------------------------===#


@always_inline
fn to_buffer[
    type: DType, rank: Int
](
    data: __mlir_type[`!pop.pointer<scalar<`, type.value, `>>`],
    shape: __mlir_type.`!pop.pointer<index>`,
) -> NDBuffer[rank, DimList[rank].create_unknown(), type]:

    let shape_scalar = __mlir_op.`pop.pointer.bitcast`[
        _type : __mlir_type.`!pop.pointer<!pop.scalar<index>>`
    ](shape)

    var shape_ptr = Pointer(shape)
    var shape_tuple = StaticIntTuple[rank]()

    var stride_tuple = StaticIntTuple[rank]()
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


@always_inline
fn elementwise_wrapper[
    trace_description: StringLiteral,
    simd_width: Int,
    type: DType,
    rank: Int,
    func: __mlir_type[
        `!kgen.signature<<`,
        Int,
        `,`,
        Int,
        `>(`,
        StaticIntTuple[
            __mlir_attr[`#kgen.param.index.ref<0, false, 1> : `, Int]
        ],
        ` borrow) capturing -> `,
        NoneType,
        `>`,
    ],
](
    buffer: NDBuffer[rank, DimList[rank].create_unknown(), type],
    out_chain: OutputChainPtr,
):
    alias unroll_factor: Int = 1

    @always_inline
    fn description_fn() -> String:
        let name_str = String("name=") + trace_description
        let shape_str = String("shape=") + String("x").join[rank](
            buffer.get_shape()
        )

        let unroll_factor_str = String("unroll_factor=") + unroll_factor
        let vector_width_str = String("vector_width=") + simd_width

        let res = String(";").join(
            name_str, shape_str, unroll_factor_str, vector_width_str
        )

        return res

    out_chain.trace_detail[TraceLevel.OP, description_fn]("mojo.elementwise")
    elementwise[rank, simd_width, unroll_factor, func](
        rebind[StaticIntTuple[rank]](buffer.dynamic_shape),
        out_chain,
    )


# ===----------------------------------------------------------------------===#
# Simd load/store helper functions
# ===----------------------------------------------------------------------===#


@always_inline
fn _compute_flat_index[
    type: DType, rank: Int, iters: Int
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
    simd_width: Int, type: DType, rank: Int
](
    buffer: NDBuffer[rank, DimList[rank].create_unknown(), type],
    index: StaticIntTuple[rank],
) -> SIMD[type, simd_width]:
    let stride = buffer.dynamic_stride[rank - 1]
    if stride == 0:
        let scalar = load_scalar[type, rank](buffer)
        return splat[type, simd_width](scalar)

    let i = stride * index[rank - 1]
    return buffer.data.simd_load[simd_width](i)


# If we know the tensor is actually a scalar tensor we can avoid all indexing
# calculation. It's broken into the two parts (load followed by splat) so we can
# hoist the load from the lambda body.
@always_inline
fn load_scalar[
    type: DType, rank: Int
](buffer: NDBuffer[rank, DimList[rank].create_unknown(), type]) -> SIMD[
    type, 1
]:
    return buffer.data.load(0)


@always_inline
fn splat[
    type: DType,
    simd_width: Int,
](val: SIMD[type, 1]) -> SIMD[type, simd_width]:
    return SIMD[type, simd_width].splat(val)


# Load a tensor which might splat along the last dimension.
@always_inline
fn simd_load_maybe_splat[
    simd_width: Int, type: DType, rank: Int
](
    buffer: NDBuffer[rank, DimList[rank].create_unknown(), type],
    index: StaticIntTuple[rank],
) -> SIMD[type, simd_width]:
    var flat_index = _compute_flat_index[type, rank, rank](buffer, index)

    if buffer.dynamic_stride[rank - 1] == 0:
        return buffer.data.load(flat_index)

    return buffer.data.simd_load[simd_width](flat_index)


# Load a tensor which does a splat along the last dimension.
@always_inline
fn simd_load_splat[
    simd_width: Int, type: DType, rank: Int
](
    buffer: NDBuffer[rank, DimList[rank].create_unknown(), type],
    index: StaticIntTuple[rank],
) -> SIMD[type, simd_width]:
    # Last dimension will be 0 for splats so don't compute last dim.
    var flat_index = _compute_flat_index[
        type, rank, rank - (1).__as_mlir_index()
    ](buffer, index)

    return buffer.data.load(flat_index)


@always_inline
fn simd_load[
    simd_width: Int, type: DType, rank: Int
](
    buffer: NDBuffer[rank, DimList[rank].create_unknown(), type],
    index: StaticIntTuple[rank],
) -> SIMD[type, simd_width]:
    let flat_index = _compute_flat_index[type, rank, rank](buffer, index)
    return buffer.data.simd_load[simd_width](flat_index)


@always_inline
fn simd_store[
    simd_width: Int, type: DType, rank: Int
](
    buffer: NDBuffer[rank, DimList[rank].create_unknown(), type],
    index: StaticIntTuple[rank],
    val: SIMD[type, simd_width],
):
    let flat_index = _compute_flat_index[type, rank, rank](buffer, index)
    buffer.data.simd_store[simd_width](flat_index, val)


# ===----------------------------------------------------------------------===#
# Broadcast
# ===----------------------------------------------------------------------===#


@always_inline
fn broadcast_to_tensor[
    type: DType,
    original_rank: Int,
    target_rank: Int,
    output_rank: Int,
](
    original: NDBuffer[
        original_rank, DimList[original_rank].create_unknown(), type
    ],
    target: NDBuffer[target_rank, DimList[target_rank].create_unknown(), type],
) -> NDBuffer[output_rank, DimList[output_rank].create_unknown(), type]:

    var shape = StaticIntTuple[output_rank]()
    var stride = StaticIntTuple[output_rank]()

    # The offset from where the implicit new dimensions end. I.E broadcasting
    # <1, 1> to <40,40,40,40> the two dimensions at the start are new
    # dimensions and then the two ones are broadcasted.
    var offset: Int = 0

    # New dimensions are always broadcast.
    @always_inline
    fn add_new_dims[i: Int]():
        @parameter
        if target_rank >= original_rank:
            shape[i] = target.dim(i)
            stride[i] = 0
        else:
            shape[i] = original.dim(i)
            stride[i] = original.stride(i)

    # Broadcast in dimensions the original started with.
    @always_inline
    fn broadcast_dim[small_index: Int]():
        # We are traversing as if they are the same size.
        let big_index = small_index + offset

        # Switch the indexes depending on which is bigger.
        var orig_index = small_index
        var target_index = big_index

        @parameter
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

    # Broadcast by appending new dimensions to the front if the sizes are not
    # the same then broadcast each of remaining dimensions. We represent this
    # using the unroll construct to help codegeneration.
    @parameter
    if target_rank < original_rank:
        unroll[original_rank - target_rank, add_new_dims]()
        offset = original_rank - target_rank
        unroll[target_rank, broadcast_dim]()
    else:
        unroll[target_rank - original_rank, add_new_dims]()
        offset = target_rank - original_rank
        unroll[original_rank, broadcast_dim]()

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
    type: DType, rank: Int
](buffer: NDBuffer[rank, DimList[rank].create_unknown(), type]):
    _printf("Rank: ")
    print(rank)
    _printf("Shape: ")
    print(buffer.dynamic_shape)

    _printf("Strides: ")
    print(buffer.dynamic_stride)
