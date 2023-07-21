# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from Assert import assert_param, debug_assert
from Buffer import Buffer, DynamicRankBuffer, NDBuffer
from BuildInfo import is_kernels_debug_build
from DType import DType
from Functional import async_parallelize
from Index import product
from Intrinsics import external_call
from List import Dim, VariadicList, DimList
from LLCL import OutputChainPtr
from Math import align_down, align_up, div_ceil, max, min
from Memory import memcpy
from Pointer import DTypePointer
from Range import range
from TargetInfo import sizeof
from Vector import InlinedFixedVector

# ===----------------------------------------------------------------------===#
# concat
# ===----------------------------------------------------------------------===#


struct _NDBufferVector[rank: Int, type: DType]:
    """Utility to store a VariadicList of NDBuffers. Required because there is not
    a clean way to convert a VariadicList of DynamicRankBuffers to a VariadicList
    of NDBuffers."""

    alias stack_capacity = 20
    alias BufferType = NDBuffer[rank, DimList.create_unknown[rank](), type]
    alias StorageType = InlinedFixedVector[Self.stack_capacity, Self.BufferType]
    var storage: Self.StorageType

    fn __init__(inout self, num_inputs: Int):
        self.storage = Self.StorageType(num_inputs)

    fn __init__(inout self, *inputs: DynamicRankBuffer):
        self.__init__(VariadicList[DynamicRankBuffer](inputs))

    fn __init__(inout self, input_list: VariadicList[DynamicRankBuffer]):
        self.storage = Self.StorageType(input_list.__len__())
        for i in range(input_list.__len__()):
            self.storage.append(input_list[i].to_ndbuffer[rank, type]())

    fn __init__(inout self, *inputs: Self.BufferType):
        self.__init__(VariadicList[Self.BufferType](inputs))

    fn __init__(inout self, input_list: VariadicList[Self.BufferType]):
        self.storage = Self.StorageType(input_list.__len__())
        for i in range(input_list.__len__()):
            self.storage.append(input_list[i])

    fn __getitem__(
        self, idx: Int
    ) -> NDBuffer[rank, DimList.create_unknown[rank](), type]:
        return self.storage[idx]

    fn __len__(self) -> Int:
        return self.storage.__len__()

    fn __del__(owned self):
        return self.storage._del_old()


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
    inputs: _NDBufferVector[rank, type],
) -> _CanonicallyReshapedBuffer:
    let input0_canon = _canonical_reshape(inputs[0], axis)
    var out_w = input0_canon.w
    for i in range(1, inputs.__len__()):
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
    inputs: _NDBufferVector[rank, type],
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

        for input_index in range(inputs.__len__()):
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

    async_parallelize[do_chunk](out_chain, num_chunks)
    external_call["KGEN_CompilerRT_LLCL_OutputChainPtr_Await", NoneType](
        out_chain.ptr
    )


@always_inline
fn _concat[
    rank: Int, type: DType
](
    output: NDBuffer[rank, DimList.create_unknown[rank](), type],
    axis: Int,
    inputs: _NDBufferVector[rank, type],
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
    for i in range(inputs.__len__()):
        w_out += inputs[i].dim(axis)

    let stride_h_out = w_out * c
    let stride_w_out = c

    var w_offset: Int = 0
    for i in range(inputs.__len__()):
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
    inputs: _NDBufferVector[rank, type],
):
    assert_param[axis == 0, "_concat_inner only supports axis 0"]()
    var num_elems_copied: Int = 0
    for i in range(inputs.__len__()):
        let buffer_len = inputs[i].size()
        memcpy[type](
            output.data.offset(num_elems_copied), inputs[i].data, buffer_len
        )
        num_elems_copied += buffer_len


fn _check_input_consistency[
    rank: Int, type: DType
](axis: Int, inputs: _NDBufferVector[rank, type],):
    @parameter
    if not is_kernels_debug_build():
        return
    # check inputs have same rank and same dims except for axis dim
    for i in range(inputs.__len__()):
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
    inputs: _NDBufferVector[rank, type],
):
    _check_input_consistency(axis, inputs)

    if axis == 0:
        _concat_inner[rank, type, 0](output, inputs)
        return

    _concat[rank, type](output, axis, inputs)


@always_inline
fn concat[
    rank: Int,
    type: DType,
](
    output: NDBuffer[rank, DimList.create_unknown[rank](), type],
    axis: Int,
    borrowed inputs: _NDBufferVector[rank, type],
    out_chain: OutputChainPtr,
):
    _check_input_consistency(axis, inputs)

    @always_inline
    @parameter
    fn dispatch_serial(unused_thread_idx: Int):
        _concat_serial[rank, type](output, axis, inputs)

    alias KB = 1024
    alias min_work_for_parallel = 128 * KB  # TODO: autotune

    let output_bytes = output.num_elements() * sizeof[type]()

    if output_bytes < min_work_for_parallel:
        async_parallelize[dispatch_serial](out_chain, 1)
        external_call["KGEN_CompilerRT_LLCL_OutputChainPtr_Await", NoneType](
            out_chain.ptr
        )
    else:
        _concat_parallel(output, axis, inputs, out_chain)
