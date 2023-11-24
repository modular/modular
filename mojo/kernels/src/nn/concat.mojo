# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from math import align_down, align_up, div_ceil, max, min
from sys import external_call
from sys._build import is_kernels_debug_build
from sys.info import simdwidthof, sizeof

from algorithm import sync_parallelize
from algorithm.functional import (
    _elementwise_impl,
    _get_start_indices_of_nth_subvolume,
)
from memory import memcpy
from memory.buffer import Buffer, NDBuffer
from memory.unsafe import DTypePointer
from MOGG import simd_load, simd_store
from runtime.llcl import OutputChainPtr

from utils.index import StaticIntTuple, product
from utils.list import Dim, DimList
from utils.vector import InlinedFixedVector

from gpu.host.memory import (
    _copy_device_to_host,
    _copy_host_to_device,
    _copy_host_to_device_async,
    _copy_device_to_device_async,
    _free,
    _malloc,
    _memset_async,
)
from gpu.host import Stream, Context, Function, synchronize
from gpu.memory import _malloc_async, _free_async

from gpu import ThreadIdx, BlockIdx

# ===----------------------------------------------------------------------===#
# concat
# ===----------------------------------------------------------------------===#


# TODO: remove once we have traits. Then, concat can take an iterable container
# which can be either a VariadicList or a FixedVector. For now, we use this to
# manually convert to a FixedVector.
fn variadic_list_to_vector[
    type: AnyRegType
](elems: VariadicList[type]) -> InlinedFixedVector[type]:
    var vector = InlinedFixedVector[type](len(elems))
    for i in range(len(elems)):
        vector.append(elems[i])
    return vector


@value
@register_passable("trivial")
struct _Span:
    var start: Int
    var end: Int

    @always_inline("nodebug")
    fn empty(self) -> Bool:
        return not (self.start < self.end)

    @always_inline("nodebug")
    fn intersect(self, other: Self) -> Self:
        return Self {
            start: max(self.start, other.start),
            end: min(self.end, other.end),
        }


@value
@register_passable("trivial")
struct _CanonicallyReshapedBuffer:
    var data: DTypePointer[DType.uint8]
    var h: Int
    var w: Int
    var c: Int


fn _canonical_reshape[
    rank: Int, type: DType
](
    buf: NDBuffer[rank, DimList.create_unknown[rank](), type], axis: Int
) -> _CanonicallyReshapedBuffer:
    let shape = buf.get_shape()
    let h = product(shape, 0, axis)
    let w = buf.dim(axis)
    let c = product(shape, axis + 1, rank) * sizeof[type]()
    return _CanonicallyReshapedBuffer(buf.data.bitcast[DType.uint8](), h, w, c)


fn _canonical_reshape_output[
    rank: Int, type: DType
](
    out_buf: NDBuffer[rank, DimList.create_unknown[rank](), type],
    axis: Int,
    inputs: InlinedFixedVector[
        NDBuffer[rank, DimList.create_unknown[rank](), type]
    ],
) -> _CanonicallyReshapedBuffer:
    let input0_canon = _canonical_reshape(inputs[0], axis)
    var out_w = input0_canon.w
    for i in range(1, len(inputs)):
        out_w += inputs[i].dim(axis)
    return _CanonicallyReshapedBuffer(
        out_buf.data.bitcast[DType.uint8](),
        input0_canon.h,
        out_w,
        input0_canon.c,
    )


fn _concat_parallel[
    rank: Int, type: DType
](
    output: NDBuffer[rank, DimList.create_unknown[rank](), type],
    axis: Int,
    inputs: InlinedFixedVector[
        NDBuffer[rank, DimList.create_unknown[rank](), type]
    ],
    out_chain: OutputChainPtr,
):
    let output_canon = _canonical_reshape_output(output, axis, inputs)

    let output_h = output_canon.h
    let output_w = output_canon.w
    let output_c = output_canon.c
    let output_wc = output_w * output_c
    let output_data = output_canon.data

    let total_output_bytes = output_h * output_wc

    alias KB = 1024
    alias parallel_chunk_size = 64 * KB  # TODO autotune
    let num_chunks = div_ceil(total_output_bytes, parallel_chunk_size)

    @parameter
    fn do_chunk(chunk_index: Int):
        # "Amount" refers to byte-offsets into logical copy order, not into
        # output buffer.
        let chunk_start_amount = chunk_index * parallel_chunk_size
        let chunk_end_amount = min(
            (chunk_index + 1) * parallel_chunk_size, total_output_bytes
        )
        let chunk_span = _Span(chunk_start_amount, chunk_end_amount)

        var amount_traversed = 0
        var output_wc_offset = 0

        for input_index in range(len(inputs)):
            let input = inputs[input_index]
            let input_canon = _canonical_reshape(input, axis)
            let input_h = input_canon.h
            let input_w = input_canon.w
            let input_c = input_canon.c
            let input_wc = input_w * input_c
            let input_data = input_canon.data
            debug_assert(input_h == output_h, "input_h != output_h")
            debug_assert(input_c == output_c, "input_c != output_c")
            let input_byte_size = input_h * input_wc

            let input_span = _Span(
                amount_traversed, amount_traversed + input_byte_size
            )
            let overlap_span = chunk_span.intersect(input_span)

            if not overlap_span.empty():
                # These are offsets of what we're trying to compute relative to
                # the start of the input buffer.
                let overlap_rel_start = overlap_span.start - input_span.start
                let overlap_rel_end = overlap_span.end - input_span.start
                # These are offsets into the input, chopping off the ends so as
                # to align to an integral 'h' index.
                let overlap_full_rel_start = align_up(
                    overlap_rel_start, input_wc
                )
                let overlap_full_rel_end = align_down(overlap_rel_end, input_wc)

                if overlap_full_rel_end < overlap_full_rel_start:
                    # If we hit here, this was probably a bad chunking choice,
                    # but let's handle it correctly anyways.
                    memcpy(
                        output_data.offset(
                            output_wc_offset
                            + overlap_rel_start // input_wc * output_wc
                            + overlap_rel_start % input_wc
                        ),
                        input_data.offset(overlap_rel_start),
                        overlap_rel_end - overlap_rel_start,
                    )
                else:
                    # OK, we have maybe stragglers on the start and end, and a
                    # nice solid middle section -- let's handle those
                    # separately.
                    # First, leading stragglers:
                    memcpy(
                        output_data.offset(
                            output_wc_offset
                            + overlap_rel_start // input_wc * output_wc
                            + overlap_rel_start % input_wc
                        ),
                        input_data.offset(overlap_rel_start),
                        overlap_full_rel_start - overlap_rel_start,
                    )
                    # Now, fully-aligned sections:
                    var in_ptr = input_data.offset(overlap_full_rel_start)
                    let end_in_ptr = input_data.offset(overlap_full_rel_end)
                    var out_ptr = output_data.offset(
                        output_wc_offset
                        + overlap_full_rel_start // input_wc * output_wc
                    )
                    while in_ptr < end_in_ptr:
                        memcpy(out_ptr, in_ptr, input_wc)
                        in_ptr += input_wc
                        out_ptr += output_wc
                    # Lastly, trailing stragglers:
                    memcpy(
                        out_ptr, in_ptr, overlap_rel_end - overlap_full_rel_end
                    )

            amount_traversed += input_byte_size
            output_wc_offset += input_wc

        debug_assert(
            amount_traversed == total_output_bytes,
            "amount_traversed != total_output_bytes",
        )

    # The do_chunk closure captures the stack allocated _NDBufferVector,
    # so this kernel must be run synchronously.
    sync_parallelize[do_chunk](out_chain, num_chunks)


@always_inline
fn _concat[
    rank: Int, type: DType
](
    output: NDBuffer[rank, DimList.create_unknown[rank](), type],
    axis: Int,
    inputs: InlinedFixedVector[
        NDBuffer[rank, DimList.create_unknown[rank](), type]
    ],
):
    """Concatenate inputs along axis and store in output.

    This simplifies the implementation by reshaping the output and inputs into 3D
    buffers. input i has dims [h, wi, c]. The output has dims [h, sum(wi), c] where
    i ranges from [0, num_inputs).

    Reshaping the buffer does not change the memory layout. After reshaping to 3D
    it is easy to visualize that the inputs can be copied in w x c sized
    contiguous slices along the h dimension.

    """

    let h = product(inputs[0].get_shape(), 0, axis)
    let c = product(inputs[0].get_shape(), axis + 1, rank)

    var w_out: Int = 0
    for i in range(len(inputs)):
        w_out += inputs[i].dim(axis)

    let stride_h_out = w_out * c
    let stride_w_out = c

    var w_offset: Int = 0
    for i in range(len(inputs)):
        # copy one w x c slice along h at a time
        let w = inputs[i].dim(axis)
        for j in range(h):
            let input_offset = j * w * c
            let output_offset = j * stride_h_out + w_offset * stride_w_out
            let in_slice = Buffer[Dim(), type](
                inputs[i].data + input_offset, w * c
            )
            let out_slice = Buffer[Dim(), type](
                output.data + output_offset, w * c
            )
            # these slices are contiguous
            memcpy(out_slice, in_slice)
        w_offset += w


@always_inline
fn _concat_inner[
    rank: Int, type: DType, axis: Int
](
    output: NDBuffer[rank, DimList.create_unknown[rank](), type],
    inputs: InlinedFixedVector[
        NDBuffer[rank, DimList.create_unknown[rank](), type]
    ],
):
    constrained[axis == 0, "_concat_inner only supports axis 0"]()
    var num_elems_copied: Int = 0
    for i in range(len(inputs)):
        let buffer_len = inputs[i].size()
        memcpy[type](
            output.data.offset(num_elems_copied), inputs[i].data, buffer_len
        )
        num_elems_copied += buffer_len


@always_inline
fn _check_input_consistency[
    rank: Int, type: DType
](
    axis: Int,
    inputs: InlinedFixedVector[
        NDBuffer[rank, DimList.create_unknown[rank](), type]
    ],
):
    @parameter
    if not is_kernels_debug_build():
        return
    # check inputs have same rank and same dims except for axis dim
    for i in range(len(inputs)):
        debug_assert(
            inputs[0].get_rank() == inputs[i].get_rank(),
            "all concat inputs must have the same rank",
        )
        for j in range(inputs[i].get_rank()):
            debug_assert(
                j == axis or inputs[0].dim(j) == inputs[i].dim(j),
                (
                    "all concat inputs must have the same dimensions in the"
                    " non-concat axes"
                ),
            )


@always_inline
fn _concat_serial[
    rank: Int, type: DType
](
    output: NDBuffer[rank, DimList.create_unknown[rank](), type],
    axis: Int,
    inputs: InlinedFixedVector[
        NDBuffer[rank, DimList.create_unknown[rank](), type]
    ],
):
    _check_input_consistency[rank, type](axis, inputs)

    if axis == 0:
        _concat_inner[rank, type, 0](output, inputs)
        return

    _concat[rank, type](output, axis, inputs)


@always_inline
fn _concat_small[
    rank: Int,
    type: DType,
](
    output: NDBuffer[rank, DimList.create_unknown[rank](), type],
    axis: Int,
    inputs: InlinedFixedVector[
        NDBuffer[rank, DimList.create_unknown[rank](), type]
    ],
    out_chain: OutputChainPtr,
):
    alias single_thread_blocking_override = True
    alias simd_width = simdwidthof[type]()

    @parameter
    @always_inline
    fn concat_lambda[
        simd_width: Int, rank: Int
    ](out_index: StaticIntTuple[rank]):
        # Concating [:, 10, :], [:, 20, :], [:, 30, :] results in shape
        # [:, 60, :] so when the target dim is:
        #   0 >= target_dim < 10: We are loading from first input.
        #   10 >= target_dim < 20: We are loading from second input.
        #   20 >= target_dim < 30: We are loading from third input.
        # The output will always be storing to the full index but we load from
        # an offset.

        var target_dim = out_index[axis]

        # Iterate through the inputs to find the one we should be storing to.
        for i in range(len(inputs)):
            let input = inputs[i]

            # This is the input we should be loading/storing.
            if target_dim < input.dynamic_shape[axis]:
                var in_index = out_index
                in_index[axis] = target_dim
                let load = simd_load[type, simd_width, rank](
                    rebind[
                        NDBuffer[rank, DimList.create_unknown[rank](), type]
                    ](input),
                    in_index,
                )
                simd_store[type, simd_width, rank](
                    rebind[
                        NDBuffer[rank, DimList.create_unknown[rank](), type]
                    ](output),
                    out_index,
                    load,
                )
                return
            else:
                # Keep looking...
                target_dim -= input.dynamic_shape[axis]

    # We need to check it's safe to simd_load from each input.
    var inputs_simd_aligned = True
    for i in range(len(inputs)):
        if inputs[i].dynamic_shape[rank - 1] % simd_width != 0:
            inputs_simd_aligned = False

    # If we are concat'ing along the last dimension we can do a simd load.
    if axis == rank - 1 and inputs_simd_aligned:
        _elementwise_impl[
            rank, simd_width, single_thread_blocking_override, concat_lambda
        ](output.dynamic_shape, out_chain)
    else:
        # Otherwise we must run scalar.
        _elementwise_impl[
            rank, 1, single_thread_blocking_override, concat_lambda
        ](output.dynamic_shape, out_chain)


@adaptive
@always_inline
fn concat[
    rank: Int,
    type: DType,
    single_thread_blocking_override: Bool,
    target: StringLiteral = "cpu",
](
    output: NDBuffer[rank, DimList.create_unknown[rank](), type],
    axis: Int,
    inputs: InlinedFixedVector[
        NDBuffer[rank, DimList.create_unknown[rank](), type]
    ],
    out_chain: OutputChainPtr,
):
    constrained[
        target == "cpu", "Concat kernel implementation only valid on CPU."
    ]()

    @parameter
    if single_thread_blocking_override:
        return _concat_small[rank, type](output, axis, inputs, out_chain)

    _check_input_consistency[rank, type](axis, inputs)

    @always_inline
    @parameter
    fn dispatch_serial(unused_thread_idx: Int):
        _concat_serial[rank, type](output, axis, inputs)

    alias KB = 1024
    alias min_work_for_parallel = 128 * KB  # TODO: autotune

    let output_bytes = output.num_elements() * sizeof[type]()

    if output_bytes < min_work_for_parallel:
        # The dispatch_serial closure captures the stack allocated
        # _NDBufferVector, so this kernel must be run synchronously.
        sync_parallelize[dispatch_serial](out_chain, 1)
    else:
        _concat_parallel(output, axis, inputs, out_chain)


@always_inline
fn concat_shape[
    input_rank: Int,
    input_type: DType,
    axis_type: DType,
    single_thread_blocking_override: Bool,
](
    input_bufs: InlinedFixedVector[
        NDBuffer[input_rank, DimList.create_unknown[input_rank](), input_type]
    ],
    axis_buf: NDBuffer[1, DimList.create_unknown[1](), axis_type],
) -> StaticIntTuple[input_rank]:
    """
    Compute the output shape of a `pad` operation, and assert the inputs are
    compatible.

    Parameters:
        input_rank: Input_rank of the input tensor.
        input_type: Type of the input tensor.
        axis_type: Type of the axis tensor.
        single_thread_blocking_override: Whether this function can block.

    Args:
        input_bufs: The input tensors list.
        axis_buf: The axis tensor.

    Returns:
        The output shape.
    """

    # extract hyper parameters
    var axis = Int__(axis_buf[0])
    if axis < 0:
        axis += input_rank
    # TODO(#17512)
    debug_assert(
        0 <= axis and axis < input_rank,
        "normalized split axis must be within range [0, input_rank)",
    )

    @always_inline
    fn shape_equal_ignore_axis(
        s1: StaticIntTuple[input_rank], s2: StaticIntTuple[input_rank]
    ) -> Bool:
        for i in range(input_rank):
            if i != axis and s1[i] != s2[i]:
                return False
        return True

    var concat_axis_dim_sum = 0
    for i in range(len(input_bufs)):
        concat_axis_dim_sum += input_bufs[i].dim(axis)
        # TODO(#17512)
        debug_assert(
            shape_equal_ignore_axis(
                input_bufs[0].get_shape(), input_bufs[i].get_shape()
            ),
            "input shapes must be equal except for at the concat axis",
        )

    # compute and return the output shape
    var output_shape = input_bufs[0].get_shape()
    output_shape[axis] = concat_axis_dim_sum
    return output_shape


@adaptive
@always_inline
fn concat[
    rank: Int,
    type: DType,
    single_thread_blocking_override: Bool,
    target: StringLiteral = "cpu",
](
    output: NDBuffer[rank, DimList.create_unknown[rank](), type],
    axis: Int,
    inputs: InlinedFixedVector[
        NDBuffer[rank, DimList.create_unknown[rank](), type]
    ],
    out_chain: OutputChainPtr,
):
    constrained[
        target == "cuda", "Concat kernel implementation only valid on GPU."
    ]()
    try:
        let num_inputs = len(inputs)
        # TODO: The number of Variadic arguments are know at compile time,
        # we should replace InlinedFixedVector with StaticTuple for any
        # operation that accept Variadic<MO_Tensor>
        if num_inputs == 1:
            return _concat_gpu[num_inputs=1](
                output,
                axis,
                StaticTuple[1](inputs[0]),
                out_chain,
            )
        if num_inputs == 2:
            return _concat_gpu[num_inputs=2](
                output,
                axis,
                StaticTuple[2](inputs[0], inputs[1]),
                out_chain,
            )
        if num_inputs == 3:
            return _concat_gpu[num_inputs=3](
                output,
                axis,
                StaticTuple[3](inputs[0], inputs[1], inputs[2]),
                out_chain,
            )
        if num_inputs == 4:
            return _concat_gpu[num_inputs=4](
                output,
                axis,
                StaticTuple[4](inputs[0], inputs[1], inputs[2], inputs[3]),
                out_chain,
            )
        if num_inputs == 5:
            return _concat_gpu[num_inputs=5](
                output,
                axis,
                StaticTuple[5](
                    inputs[0], inputs[1], inputs[2], inputs[3], inputs[4]
                ),
                out_chain,
            )
        if num_inputs == 6:
            return _concat_gpu[num_inputs=6](
                output,
                axis,
                StaticTuple[6](
                    inputs[0],
                    inputs[1],
                    inputs[2],
                    inputs[3],
                    inputs[4],
                    inputs[5],
                ),
                out_chain,
            )
        if num_inputs == 7:
            return _concat_gpu[num_inputs=7](
                output,
                axis,
                StaticTuple[7](
                    inputs[0],
                    inputs[1],
                    inputs[2],
                    inputs[3],
                    inputs[4],
                    inputs[5],
                    inputs[6],
                ),
                out_chain,
            )
        if num_inputs == 8:
            return _concat_gpu[num_inputs=8](
                output,
                axis,
                StaticTuple[8](
                    inputs[0],
                    inputs[1],
                    inputs[2],
                    inputs[3],
                    inputs[4],
                    inputs[5],
                    inputs[6],
                    inputs[7],
                ),
                out_chain,
            )
        if num_inputs == 9:
            return _concat_gpu[num_inputs=9](
                output,
                axis,
                StaticTuple[9](
                    inputs[0],
                    inputs[1],
                    inputs[2],
                    inputs[3],
                    inputs[4],
                    inputs[5],
                    inputs[6],
                    inputs[7],
                    inputs[8],
                ),
                out_chain,
            )
        if num_inputs == 10:
            return _concat_gpu[num_inputs=10](
                output,
                axis,
                StaticTuple[10](
                    inputs[0],
                    inputs[1],
                    inputs[2],
                    inputs[3],
                    inputs[4],
                    inputs[5],
                    inputs[6],
                    inputs[7],
                    inputs[8],
                    inputs[9],
                ),
                out_chain,
            )
        if num_inputs == 11:
            return _concat_gpu[num_inputs=11](
                output,
                axis,
                StaticTuple[11](
                    inputs[0],
                    inputs[1],
                    inputs[2],
                    inputs[3],
                    inputs[4],
                    inputs[5],
                    inputs[6],
                    inputs[7],
                    inputs[8],
                    inputs[9],
                    inputs[10],
                ),
                out_chain,
            )
        if num_inputs == 12:
            return _concat_gpu[num_inputs=12](
                output,
                axis,
                StaticTuple[12](
                    inputs[0],
                    inputs[1],
                    inputs[2],
                    inputs[3],
                    inputs[4],
                    inputs[5],
                    inputs[6],
                    inputs[7],
                    inputs[8],
                    inputs[9],
                    inputs[10],
                    inputs[11],
                ),
                out_chain,
            )
        if num_inputs == 13:
            return _concat_gpu[num_inputs=13](
                output,
                axis,
                StaticTuple[13](
                    inputs[0],
                    inputs[1],
                    inputs[2],
                    inputs[3],
                    inputs[4],
                    inputs[5],
                    inputs[6],
                    inputs[7],
                    inputs[8],
                    inputs[9],
                    inputs[10],
                    inputs[11],
                    inputs[12],
                ),
                out_chain,
            )
        if num_inputs == 14:
            return _concat_gpu[num_inputs=14](
                output,
                axis,
                StaticTuple[14](
                    inputs[0],
                    inputs[1],
                    inputs[2],
                    inputs[3],
                    inputs[4],
                    inputs[5],
                    inputs[6],
                    inputs[7],
                    inputs[8],
                    inputs[9],
                    inputs[10],
                    inputs[11],
                    inputs[12],
                    inputs[13],
                ),
                out_chain,
            )
        if num_inputs == 15:
            return _concat_gpu[num_inputs=15](
                output,
                axis,
                StaticTuple[15](
                    inputs[0],
                    inputs[1],
                    inputs[2],
                    inputs[3],
                    inputs[4],
                    inputs[5],
                    inputs[6],
                    inputs[7],
                    inputs[8],
                    inputs[9],
                    inputs[10],
                    inputs[11],
                    inputs[12],
                    inputs[13],
                    inputs[14],
                ),
                out_chain,
            )
        if num_inputs == 16:
            return _concat_gpu[num_inputs=16](
                output,
                axis,
                StaticTuple[16](
                    inputs[0],
                    inputs[1],
                    inputs[2],
                    inputs[3],
                    inputs[4],
                    inputs[5],
                    inputs[6],
                    inputs[7],
                    inputs[8],
                    inputs[9],
                    inputs[10],
                    inputs[11],
                    inputs[12],
                    inputs[13],
                    inputs[14],
                    inputs[15],
                ),
                out_chain,
            )
        if num_inputs == 17:
            return _concat_gpu[num_inputs=17](
                output,
                axis,
                StaticTuple[17](
                    inputs[0],
                    inputs[1],
                    inputs[2],
                    inputs[3],
                    inputs[4],
                    inputs[5],
                    inputs[6],
                    inputs[7],
                    inputs[8],
                    inputs[9],
                    inputs[10],
                    inputs[11],
                    inputs[12],
                    inputs[13],
                    inputs[14],
                    inputs[15],
                    inputs[16],
                ),
                out_chain,
            )
        if num_inputs == 18:
            return _concat_gpu[num_inputs=18](
                output,
                axis,
                StaticTuple[18](
                    inputs[0],
                    inputs[1],
                    inputs[2],
                    inputs[3],
                    inputs[4],
                    inputs[5],
                    inputs[6],
                    inputs[7],
                    inputs[8],
                    inputs[9],
                    inputs[10],
                    inputs[11],
                    inputs[12],
                    inputs[13],
                    inputs[14],
                    inputs[15],
                    inputs[16],
                    inputs[17],
                ),
                out_chain,
            )
        if num_inputs == 19:
            return _concat_gpu[num_inputs=19](
                output,
                axis,
                StaticTuple[19](
                    inputs[0],
                    inputs[1],
                    inputs[2],
                    inputs[3],
                    inputs[4],
                    inputs[5],
                    inputs[6],
                    inputs[7],
                    inputs[8],
                    inputs[9],
                    inputs[10],
                    inputs[11],
                    inputs[12],
                    inputs[13],
                    inputs[14],
                    inputs[15],
                    inputs[16],
                    inputs[17],
                    inputs[18],
                ),
                out_chain,
            )
        if num_inputs == 20:
            return _concat_gpu[num_inputs=20](
                output,
                axis,
                StaticTuple[20](
                    inputs[0],
                    inputs[1],
                    inputs[2],
                    inputs[3],
                    inputs[4],
                    inputs[5],
                    inputs[6],
                    inputs[7],
                    inputs[8],
                    inputs[9],
                    inputs[10],
                    inputs[11],
                    inputs[12],
                    inputs[13],
                    inputs[14],
                    inputs[15],
                    inputs[16],
                    inputs[17],
                    inputs[18],
                    inputs[19],
                ),
                out_chain,
            )
        if num_inputs == 21:
            return _concat_gpu[num_inputs=21](
                output,
                axis,
                StaticTuple[21](
                    inputs[0],
                    inputs[1],
                    inputs[2],
                    inputs[3],
                    inputs[4],
                    inputs[5],
                    inputs[6],
                    inputs[7],
                    inputs[8],
                    inputs[9],
                    inputs[10],
                    inputs[11],
                    inputs[12],
                    inputs[13],
                    inputs[14],
                    inputs[15],
                    inputs[16],
                    inputs[17],
                    inputs[18],
                    inputs[19],
                    inputs[20],
                ),
                out_chain,
            )
        if num_inputs == 22:
            return _concat_gpu[num_inputs=22](
                output,
                axis,
                StaticTuple[22](
                    inputs[0],
                    inputs[1],
                    inputs[2],
                    inputs[3],
                    inputs[4],
                    inputs[5],
                    inputs[6],
                    inputs[7],
                    inputs[8],
                    inputs[9],
                    inputs[10],
                    inputs[11],
                    inputs[12],
                    inputs[13],
                    inputs[14],
                    inputs[15],
                    inputs[16],
                    inputs[17],
                    inputs[18],
                    inputs[19],
                    inputs[20],
                    inputs[21],
                ),
                out_chain,
            )
        if num_inputs == 23:
            return _concat_gpu[num_inputs=23](
                output,
                axis,
                StaticTuple[23](
                    inputs[0],
                    inputs[1],
                    inputs[2],
                    inputs[3],
                    inputs[4],
                    inputs[5],
                    inputs[6],
                    inputs[7],
                    inputs[8],
                    inputs[9],
                    inputs[10],
                    inputs[11],
                    inputs[12],
                    inputs[13],
                    inputs[14],
                    inputs[15],
                    inputs[16],
                    inputs[17],
                    inputs[18],
                    inputs[19],
                    inputs[20],
                    inputs[21],
                    inputs[22],
                ),
                out_chain,
            )
        if num_inputs == 24:
            return _concat_gpu[num_inputs=24](
                output,
                axis,
                StaticTuple[24](
                    inputs[0],
                    inputs[1],
                    inputs[2],
                    inputs[3],
                    inputs[4],
                    inputs[5],
                    inputs[6],
                    inputs[7],
                    inputs[8],
                    inputs[9],
                    inputs[10],
                    inputs[11],
                    inputs[12],
                    inputs[13],
                    inputs[14],
                    inputs[15],
                    inputs[16],
                    inputs[17],
                    inputs[18],
                    inputs[19],
                    inputs[20],
                    inputs[21],
                    inputs[22],
                    inputs[23],
                ),
                out_chain,
            )
        if num_inputs == 25:
            return _concat_gpu[num_inputs=25](
                output,
                axis,
                StaticTuple[25](
                    inputs[0],
                    inputs[1],
                    inputs[2],
                    inputs[3],
                    inputs[4],
                    inputs[5],
                    inputs[6],
                    inputs[7],
                    inputs[8],
                    inputs[9],
                    inputs[10],
                    inputs[11],
                    inputs[12],
                    inputs[13],
                    inputs[14],
                    inputs[15],
                    inputs[16],
                    inputs[17],
                    inputs[18],
                    inputs[19],
                    inputs[20],
                    inputs[21],
                    inputs[22],
                    inputs[23],
                    inputs[24],
                ),
                out_chain,
            )
        if num_inputs == 26:
            return _concat_gpu[num_inputs=26](
                output,
                axis,
                StaticTuple[26](
                    inputs[0],
                    inputs[1],
                    inputs[2],
                    inputs[3],
                    inputs[4],
                    inputs[5],
                    inputs[6],
                    inputs[7],
                    inputs[8],
                    inputs[9],
                    inputs[10],
                    inputs[11],
                    inputs[12],
                    inputs[13],
                    inputs[14],
                    inputs[15],
                    inputs[16],
                    inputs[17],
                    inputs[18],
                    inputs[19],
                    inputs[20],
                    inputs[21],
                    inputs[22],
                    inputs[23],
                    inputs[24],
                    inputs[25],
                ),
                out_chain,
            )
        if num_inputs == 27:
            return _concat_gpu[num_inputs=27](
                output,
                axis,
                StaticTuple[27](
                    inputs[0],
                    inputs[1],
                    inputs[2],
                    inputs[3],
                    inputs[4],
                    inputs[5],
                    inputs[6],
                    inputs[7],
                    inputs[8],
                    inputs[9],
                    inputs[10],
                    inputs[11],
                    inputs[12],
                    inputs[13],
                    inputs[14],
                    inputs[15],
                    inputs[16],
                    inputs[17],
                    inputs[18],
                    inputs[19],
                    inputs[20],
                    inputs[21],
                    inputs[22],
                    inputs[23],
                    inputs[24],
                    inputs[25],
                    inputs[26],
                ),
                out_chain,
            )
        if num_inputs == 28:
            return _concat_gpu[num_inputs=28](
                output,
                axis,
                StaticTuple[28](
                    inputs[0],
                    inputs[1],
                    inputs[2],
                    inputs[3],
                    inputs[4],
                    inputs[5],
                    inputs[6],
                    inputs[7],
                    inputs[8],
                    inputs[9],
                    inputs[10],
                    inputs[11],
                    inputs[12],
                    inputs[13],
                    inputs[14],
                    inputs[15],
                    inputs[16],
                    inputs[17],
                    inputs[18],
                    inputs[19],
                    inputs[20],
                    inputs[21],
                    inputs[22],
                    inputs[23],
                    inputs[24],
                    inputs[25],
                    inputs[26],
                    inputs[27],
                ),
                out_chain,
            )
        if num_inputs == 29:
            return _concat_gpu[num_inputs=29](
                output,
                axis,
                StaticTuple[29](
                    inputs[0],
                    inputs[1],
                    inputs[2],
                    inputs[3],
                    inputs[4],
                    inputs[5],
                    inputs[6],
                    inputs[7],
                    inputs[8],
                    inputs[9],
                    inputs[10],
                    inputs[11],
                    inputs[12],
                    inputs[13],
                    inputs[14],
                    inputs[15],
                    inputs[16],
                    inputs[17],
                    inputs[18],
                    inputs[19],
                    inputs[20],
                    inputs[21],
                    inputs[22],
                    inputs[23],
                    inputs[24],
                    inputs[25],
                    inputs[26],
                    inputs[27],
                    inputs[28],
                ),
                out_chain,
            )
        if num_inputs == 30:
            return _concat_gpu[num_inputs=30](
                output,
                axis,
                StaticTuple[30](
                    inputs[0],
                    inputs[1],
                    inputs[2],
                    inputs[3],
                    inputs[4],
                    inputs[5],
                    inputs[6],
                    inputs[7],
                    inputs[8],
                    inputs[9],
                    inputs[10],
                    inputs[11],
                    inputs[12],
                    inputs[13],
                    inputs[14],
                    inputs[15],
                    inputs[16],
                    inputs[17],
                    inputs[18],
                    inputs[19],
                    inputs[20],
                    inputs[21],
                    inputs[22],
                    inputs[23],
                    inputs[24],
                    inputs[25],
                    inputs[26],
                    inputs[27],
                    inputs[28],
                    inputs[29],
                ),
                out_chain,
            )
        if num_inputs == 31:
            return _concat_gpu[num_inputs=31](
                output,
                axis,
                StaticTuple[31](
                    inputs[0],
                    inputs[1],
                    inputs[2],
                    inputs[3],
                    inputs[4],
                    inputs[5],
                    inputs[6],
                    inputs[7],
                    inputs[8],
                    inputs[9],
                    inputs[10],
                    inputs[11],
                    inputs[12],
                    inputs[13],
                    inputs[14],
                    inputs[15],
                    inputs[16],
                    inputs[17],
                    inputs[18],
                    inputs[19],
                    inputs[20],
                    inputs[21],
                    inputs[22],
                    inputs[23],
                    inputs[24],
                    inputs[25],
                    inputs[26],
                    inputs[27],
                    inputs[28],
                    inputs[29],
                    inputs[30],
                ),
                out_chain,
            )
        if num_inputs == 32:
            return _concat_gpu[num_inputs=32](
                output,
                axis,
                StaticTuple[32](
                    inputs[0],
                    inputs[1],
                    inputs[2],
                    inputs[3],
                    inputs[4],
                    inputs[5],
                    inputs[6],
                    inputs[7],
                    inputs[8],
                    inputs[9],
                    inputs[10],
                    inputs[11],
                    inputs[12],
                    inputs[13],
                    inputs[14],
                    inputs[15],
                    inputs[16],
                    inputs[17],
                    inputs[18],
                    inputs[19],
                    inputs[20],
                    inputs[21],
                    inputs[22],
                    inputs[23],
                    inputs[24],
                    inputs[25],
                    inputs[26],
                    inputs[27],
                    inputs[28],
                    inputs[29],
                    inputs[30],
                    inputs[31],
                ),
                out_chain,
            )

        else:
            return out_chain.mark_error(
                "Unsupported concat with num_inputs > 32"
            )
    except e:
        return out_chain.mark_error(e)


fn _concat_inner_most_single_dim[
    rank: Int, type: DType, num_inputs: Int, block_size: Int
](
    output: NDBuffer[rank, DimList.create_unknown[rank](), type],
    inputs: StaticTuple[
        num_inputs, NDBuffer[rank, DimList.create_unknown[rank](), type]
    ],
):
    let idx = BlockIdx.x() * block_size + ThreadIdx.x()
    let index = _get_start_indices_of_nth_subvolume[rank, 1](
        idx, output.dynamic_shape
    )

    if index > output.num_elements():
        return

    @unroll
    for i in range(num_inputs):
        var out_index = index
        out_index[rank - 1] = i
        output[out_index] = inputs[i][index]


@always_inline
fn _concat_gpu[
    rank: Int,
    type: DType,
    num_inputs: Int,
](
    output: NDBuffer[rank, DimList.create_unknown[rank](), type],
    axis: Int,
    inputs: StaticTuple[
        num_inputs, NDBuffer[rank, DimList.create_unknown[rank](), type]
    ],
    out_chain: OutputChainPtr,
) raises:
    var stream = out_chain.get_cuda_stream() if out_chain else Stream[
        is_borrowed=True
    ]()

    # Size of outer dims, if 1 we should memcpy to the output buffer.
    var outer_dims = 1
    for i in range(axis):
        # Use input[0], all dims should be equal except axis.
        outer_dims *= inputs[0].dim(i)

    @parameter
    @always_inline
    fn _concat_buffers_contiguously() raises:
        var input_size = 0

        @unroll
        for i in range(num_inputs):
            # Skip empty inputs.
            if inputs[i].num_elements() > 0:
                _copy_device_to_device_async(
                    output.data.offset(input_size),
                    inputs[i].data,
                    inputs[i].num_elements(),
                    stream,
                )
                input_size += inputs[i].num_elements()

    # If outer_dims are ones use device-to-device copies.
    if outer_dims == 1:
        return _concat_buffers_contiguously()

    @parameter
    @always_inline
    fn per_output_elem[
        simd_width: Int, _rank: Int
    ](out_index: StaticIntTuple[_rank]):
        var in_index = out_index
        in_index[axis] = out_index[axis]

        # can use binary search here to reduce num iters to log2(num_inputs)
        for i in range(num_inputs):
            let input = inputs[i]

            if in_index[axis] < input.get_shape()[axis]:
                output[rebind[StaticIntTuple[rank]](out_index)] = input[
                    rebind[StaticIntTuple[rank]](in_index)
                ]
                return

            in_index[axis] -= input.get_shape()[axis]

    if axis == rank - 1:
        var inner_most_unit_dim = True
        for i in range(num_inputs):
            if inputs[i].dim(axis) != 1:
                inner_most_unit_dim = False
                break

        if inner_most_unit_dim:
            alias block_size = 32
            let func = Function[
                fn (
                    NDBuffer[rank, DimList.create_unknown[rank](), type],
                    StaticTuple[
                        num_inputs,
                        NDBuffer[rank, DimList.create_unknown[rank](), type],
                    ],
                ) -> None, _concat_inner_most_single_dim[
                    rank, type, num_inputs, block_size
                ]
            ]()

            return func(
                (output.num_elements() // block_size),
                (block_size),
                output,
                inputs,
                stream=stream,
            )

    # Can picture output reshaped to 3D: output_reshape = reshape(output, dims=[-1, concat_dim, -1])
    # where concat_dim = inputs[0][axis] + ... + inputs[n-1][axis].
    # Slices of the innermost dim of output_reshape are contiguous in the corresponding input.
    # Because the inner dim is contiguous we will get coalesced memory access
    # using the elementwise generator with simd_width=1.
    alias target = "cuda"
    _elementwise_impl[
        rank,
        1,
        False,
        per_output_elem,
        target,
    ](output.get_shape(), out_chain)
