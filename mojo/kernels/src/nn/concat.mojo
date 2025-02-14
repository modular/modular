# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from collections import OptionalReg
from collections.vector import InlinedFixedVector
from math import align_down, align_up, ceildiv
from sys._build import is_debug_build
from sys.info import bitwidthof, simdwidthof, sizeof

from algorithm.functional import (
    _get_start_indices_of_nth_subvolume,
    _get_start_indices_of_nth_subvolume_uint,
    elementwise,
    sync_parallelize,
)
from buffer import NDBuffer
from gpu import block_idx, thread_idx
from gpu.host import DeviceBuffer, DeviceContext
from gpu.host.info import is_cpu, is_valid_target
from memory import UnsafePointer, memcpy
from register import register_internal
from runtime.asyncrt import MojoCallContextPtr
from runtime.tracing import Trace, TraceLevel

from utils import IndexList, StaticTuple, product

alias elementwise_epilogue_type = fn[
    c_type: DType, rank: Int, width: Int = 1, *, alignment: Int = 1
] (IndexList[rank], SIMD[c_type, width]) capturing -> None


# ===-----------------------------------------------------------------------===#
# concat
# ===-----------------------------------------------------------------------===#


@always_inline
fn memcpy_or_fuse[
    rank: Int,
    type: DType,
    epilogue_fn: OptionalReg[elementwise_epilogue_type],
](
    dest_data: UnsafePointer[Int8],
    out_byte_offset: Int,
    src_data: __type_of(dest_data),
    n: Int,
    out_shape: IndexList[rank, **_],
):
    @parameter
    if not epilogue_fn:
        memcpy(dest_data.offset(out_byte_offset), src_data, n)
    else:
        alias func = epilogue_fn.value()
        alias simd_width = simdwidthof[type]()

        var typed_offset = out_byte_offset // sizeof[type]()
        var typed_len = n // sizeof[type]()
        debug_assert(
            n % sizeof[type]() == 0 and out_byte_offset % sizeof[type]() == 0,
            "offset and length must be dividable by sizeof[type]",
        )

        # Cast
        var shape_1d = IndexList[1](typed_len)
        var typed_src = src_data.bitcast[Scalar[type]]()
        var input = NDBuffer[type, 1](
            typed_src,
            shape_1d,
        )

        @parameter
        @always_inline
        fn epilogue_wrapper[
            simd_width: Int, _rank: Int
        ](index: IndexList[_rank]):
            var load = rebind[NDBuffer[type, _rank]](input).load[
                width=simd_width
            ](index)

            # Convert the linearized address back to the nd indices.
            constrained[_rank == 1]()
            var out_index = _get_start_indices_of_nth_subvolume[0](
                index[0] + typed_offset,
                out_shape,
            )

            func[type, rank, simd_width](
                out_index.cast[
                    element_bitwidth = bitwidthof[Int](), unsigned=False
                ](),
                load,
            )
            return

        # We must run scalar to be conservative. This is because the fused
        # output lambda might operate on views (e.g., broadcast) that does not
        # always work with indices produced from a linearized address.
        elementwise[epilogue_wrapper, simd_width=1](shape_1d)


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
        return Self(max(self.start, other.start), min(self.end, other.end))


@value
@register_passable("trivial")
struct _CanonicallyReshapedBuffer:
    var data: UnsafePointer[Int8]
    var h: Int
    var w: Int
    var c: Int


fn _canonical_reshape[
    rank: Int, type: DType
](buf: NDBuffer[type, rank], axis: Int) -> _CanonicallyReshapedBuffer:
    var shape = buf.get_shape()
    var h = product(shape, 0, axis)
    var w = buf.dim(axis)
    var c = product(shape, axis + 1, rank) * sizeof[type]()
    return _CanonicallyReshapedBuffer(buf.data.bitcast[Int8](), h, w, c)


fn _canonical_reshape_output[
    rank: Int, type: DType
](
    out_buf: NDBuffer[type, rank],
    axis: Int,
    inputs: InlinedFixedVector[NDBuffer[type, rank]],
) -> _CanonicallyReshapedBuffer:
    var input0_canon = _canonical_reshape(inputs[0], axis)
    var out_w = input0_canon.w
    for i in range(1, len(inputs)):
        out_w += inputs[i].dim(axis)
    return _CanonicallyReshapedBuffer(
        out_buf.data.bitcast[Int8](),
        input0_canon.h,
        out_w,
        input0_canon.c,
    )


fn _concat_parallel[
    rank: Int,
    type: DType,
    epilogue_fn: OptionalReg[elementwise_epilogue_type],
](
    output: NDBuffer[type, rank],
    axis: Int,
    inputs: InlinedFixedVector[NDBuffer[type, rank]],
):
    var output_canon = _canonical_reshape_output(output, axis, inputs)

    var output_h = output_canon.h
    var output_w = output_canon.w
    var output_c = output_canon.c
    var output_wc = output_w * output_c
    var output_data = output_canon.data

    var total_output_bytes = output_h * output_wc

    alias KB = 1024
    alias parallel_chunk_size = 64 * KB  # TODO autotune
    var num_chunks = ceildiv(total_output_bytes, parallel_chunk_size)

    @__copy_capture(
        total_output_bytes, output_h, output_c, output_data, output_wc
    )
    @parameter
    fn do_chunk(chunk_index: Int):
        # "Amount" refers to byte-offsets into logical copy order, not into
        # output buffer.
        var chunk_start_amount = chunk_index * parallel_chunk_size
        var chunk_end_amount = min(
            (chunk_index + 1) * parallel_chunk_size, total_output_bytes
        )
        var chunk_span = _Span(chunk_start_amount, chunk_end_amount)

        var amount_traversed = 0
        var output_wc_offset = 0

        for input_index in range(len(inputs)):
            var input = inputs[input_index]
            var input_canon = _canonical_reshape(input, axis)
            var input_h = input_canon.h
            var input_w = input_canon.w
            var input_c = input_canon.c
            var input_wc = input_w * input_c
            var input_data = input_canon.data
            debug_assert(input_h == output_h, "input_h != output_h")
            debug_assert(input_c == output_c, "input_c != output_c")
            var input_byte_size = input_h * input_wc

            var input_span = _Span(
                amount_traversed, amount_traversed + input_byte_size
            )
            var overlap_span = chunk_span.intersect(input_span)

            if not overlap_span.empty():
                # These are offsets of what we're trying to compute relative to
                # the start of the input buffer.
                var overlap_rel_start = overlap_span.start - input_span.start
                var overlap_rel_end = overlap_span.end - input_span.start
                # These are offsets into the input, chopping off the ends so as
                # to align to an integral 'h' index.
                var overlap_full_rel_start = align_up(
                    overlap_rel_start, input_wc
                )
                var overlap_full_rel_end = align_down(overlap_rel_end, input_wc)

                if overlap_full_rel_end < overlap_full_rel_start:
                    # If we hit here, this was probably a bad chunking choice,
                    # but var's handle it correctly anyways.
                    memcpy_or_fuse[rank, type, epilogue_fn](
                        output_data,
                        output_wc_offset
                        + overlap_rel_start // input_wc * output_wc
                        + overlap_rel_start % input_wc,
                        input_data.offset(overlap_rel_start),
                        overlap_rel_end - overlap_rel_start,
                        output.dynamic_shape,
                    )
                else:
                    # OK, we have maybe stragglers on the start and end, and a
                    # nice solid middle section -- var's handle those
                    # separately.
                    # First, leading stragglers:
                    memcpy_or_fuse[rank, type, epilogue_fn](
                        output_data,
                        output_wc_offset
                        + overlap_rel_start // input_wc * output_wc
                        + overlap_rel_start % input_wc,
                        input_data.offset(overlap_rel_start),
                        overlap_full_rel_start - overlap_rel_start,
                        output.dynamic_shape,
                    )
                    # Now, fully-aligned sections:
                    var in_ptr = input_data.offset(overlap_full_rel_start)
                    var end_in_ptr = input_data.offset(overlap_full_rel_end)
                    var out_ptr_offset = output_wc_offset + overlap_full_rel_start // input_wc * output_wc

                    while in_ptr < end_in_ptr:
                        memcpy_or_fuse[rank, type, epilogue_fn](
                            output_data,
                            out_ptr_offset,
                            in_ptr,
                            input_wc,
                            output.dynamic_shape,
                        )
                        in_ptr += input_wc
                        out_ptr_offset += output_wc
                    # Lastly, trailing stragglers:
                    memcpy_or_fuse[rank, type, epilogue_fn](
                        output_data,
                        out_ptr_offset,
                        in_ptr,
                        overlap_rel_end - overlap_full_rel_end,
                        output.dynamic_shape,
                    )

            amount_traversed += input_byte_size
            output_wc_offset += input_wc

        debug_assert(
            amount_traversed == total_output_bytes,
            "amount_traversed != total_output_bytes",
        )

    # The do_chunk closure captures the stack allocated Buffer,
    # so this kernel must be run synchronously.
    sync_parallelize[do_chunk](num_chunks)


@always_inline
fn _concat[
    rank: Int,
    type: DType,
    epilogue_fn: OptionalReg[elementwise_epilogue_type],
](
    output: NDBuffer[type, rank],
    axis: Int,
    inputs: InlinedFixedVector[NDBuffer[type, rank]],
):
    """Concatenate inputs along axis and store in output.

    This simplifies the implementation by reshaping the output and inputs into 3D
    buffers. input i has dims [h, wi, c]. The output has dims [h, sum(wi), c] where
    i ranges from [0, num_inputs).

    Reshaping the buffer does not change the memory layout. After reshaping to 3D
    it is easy to visualize that the inputs can be copied in w x c sized
    contiguous slices along the h dimension.

    """

    var h = product(inputs[0].get_shape(), 0, axis)
    var c = product(inputs[0].get_shape(), axis + 1, rank)

    var w_out: Int = 0
    for i in range(len(inputs)):
        w_out += inputs[i].dim(axis)

    var stride_h_out = w_out * c
    var stride_w_out = c

    var w_offset: Int = 0
    for i in range(len(inputs)):
        # copy one w x c slice along h at a time
        var w = inputs[i].dim(axis)
        for j in range(h):
            var input_offset = j * w * c
            var output_offset = j * stride_h_out + w_offset * stride_w_out
            # these slices are contiguous
            memcpy_or_fuse[rank, type, epilogue_fn](
                output.data.bitcast[Int8](),
                output_offset * sizeof[type](),
                (inputs[i].data + input_offset).bitcast[Int8](),
                w * c * sizeof[type](),
                output.dynamic_shape,
            )
        w_offset += w


@always_inline
fn _concat_inner[
    rank: Int,
    type: DType,
    epilogue_fn: OptionalReg[elementwise_epilogue_type],
](
    output: NDBuffer[type, rank],
    inputs: InlinedFixedVector[NDBuffer[type, rank]],
):
    var num_elems_copied: Int = 0
    for i in range(len(inputs)):
        var buffer_len = inputs[i].size()
        memcpy_or_fuse[rank, type, epilogue_fn](
            output.data.bitcast[Int8](),
            num_elems_copied * sizeof[type](),
            inputs[i].data.bitcast[Int8](),
            buffer_len * sizeof[type](),
            output.dynamic_shape,
        )
        num_elems_copied += buffer_len


@always_inline
fn _check_input_consistency[
    rank: Int, type: DType
](axis: Int, inputs: InlinedFixedVector[NDBuffer[type, rank]],):
    @parameter
    if not is_debug_build():
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
    rank: Int,
    type: DType,
    epilogue_fn: OptionalReg[elementwise_epilogue_type],
](
    output: NDBuffer[type, rank],
    axis: Int,
    inputs: InlinedFixedVector[NDBuffer[type, rank]],
):
    _check_input_consistency[rank, type](axis, inputs)

    var all_outer_dims_singvaron = True
    for i in range(axis):
        if inputs[0].dim(i) == 1:
            continue

        all_outer_dims_singvaron = False
        break

    if all_outer_dims_singvaron:
        _concat_inner[rank, type, epilogue_fn](output, inputs)
        return

    _concat[rank, type, epilogue_fn](output, axis, inputs)


@always_inline
fn _concat_small[
    rank: Int,
    type: DType,
    epilogue_fn: OptionalReg[elementwise_epilogue_type],
](
    output: NDBuffer[type, rank],
    axis: Int,
    inputs: InlinedFixedVector[NDBuffer[type, rank]],
):
    alias single_thread_blocking_override = True
    alias simd_width = simdwidthof[type]()

    @parameter
    @always_inline
    fn concat_lambda[simd_width: Int, rank: Int](out_index: IndexList[rank]):
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
            var input = inputs[i]
            # This is the input we should be loading/storing.
            if target_dim < input.dynamic_shape[axis]:
                var in_index = out_index
                in_index[axis] = target_dim
                var load = rebind[NDBuffer[type, rank]](input).load[
                    width=simd_width
                ](in_index)

                @parameter
                if epilogue_fn:
                    alias func = epilogue_fn.value()
                    func[type, rank, simd_width](out_index, load)
                else:
                    rebind[NDBuffer[type, rank]](output).store[
                        width=simd_width
                    ](out_index, load)
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
        elementwise[
            concat_lambda,
            simd_width=simd_width,
            use_blocking_impl=single_thread_blocking_override,
        ](output.dynamic_shape)
    else:
        # Otherwise we must run scalar.
        elementwise[
            concat_lambda,
            simd_width=1,
            use_blocking_impl=single_thread_blocking_override,
        ](output.dynamic_shape)


@always_inline
fn _concat_cpu[
    rank: Int,
    type: DType,
    epilogue_fn: OptionalReg[elementwise_epilogue_type],
    single_thread_blocking_override: Bool,
](
    output: NDBuffer[type, rank],
    axis: Int,
    inputs: InlinedFixedVector[NDBuffer[type, rank]],
) raises:
    @parameter
    if single_thread_blocking_override:
        return _concat_small[rank, type, epilogue_fn](output, axis, inputs)

    _check_input_consistency[rank, type](axis, inputs)

    @always_inline
    @parameter
    fn dispatch_serial(unused_thread_idx: Int):
        _concat_serial[rank, type, epilogue_fn](output, axis, inputs)

    alias KB = 1024
    alias min_work_for_parallel = 128 * KB  # TODO: autotune

    var output_bytes = output.num_elements() * sizeof[type]()

    if output_bytes < min_work_for_parallel:
        # The dispatch_serial closure captures the stack allocated
        # Buffer, so this kernel must be run synchronously.
        sync_parallelize[dispatch_serial](1)
    else:
        _concat_parallel[epilogue_fn=epilogue_fn](output, axis, inputs)


@always_inline
fn concat_shape[
    input_rank: Int,
    input_type: DType,
    axis_type: DType,
    single_thread_blocking_override: Bool,
](
    input_bufs: InlinedFixedVector[NDBuffer[input_type, input_rank]],
    axis_buf: NDBuffer[axis_type, 1],
) raises -> IndexList[input_rank]:
    """
    Compute the output shape of a `pad` operation, and assert the inputs are
    compatible.

    Parameters:
        input_rank: Input_rank of the input tensor.
        input_type: Type of the input tensor.
        axis_type: Type of the axis tensor.
        single_thread_blocking_override: If True, then the operation is run
          synchronously using a single thread.

    Args:
        input_bufs: The input tensors list.
        axis_buf: The axis tensor.

    Returns:
        The output shape.
    """

    # extract hyper parameters
    var axis = Int(axis_buf[0])
    if axis < 0:
        axis += input_rank
    if axis < 0 or input_rank <= axis:
        raise Error(
            "[concat_from_list] normalized axis must be within range [0,"
            " input_rank)"
        )

    @parameter
    @always_inline
    fn shape_equal_ignore_axis(
        s1: IndexList[input_rank], s2: IndexList[input_rank]
    ) -> Bool:
        for i in range(input_rank):
            if i != axis and s1[i] != s2[i]:
                return False
        return True

    var concat_axis_dim_sum = 0
    for i in range(len(input_bufs)):
        concat_axis_dim_sum += input_bufs[i].dim(axis)
        if not shape_equal_ignore_axis(
            input_bufs[0].get_shape(), input_bufs[i].get_shape()
        ):
            raise Error(
                "[concat_from_list] input shapes must match except at concat"
                " axis"
            )

    # compute and return the output shape
    var output_shape = input_bufs[0].get_shape()
    output_shape[axis] = concat_axis_dim_sum
    return output_shape


@always_inline
fn concat[
    rank: Int,
    type: DType,
    single_thread_blocking_override: Bool,
    target: StringLiteral = "cpu",
    epilogue_fn: OptionalReg[elementwise_epilogue_type] = None,
](
    output: NDBuffer[type, rank],
    axis: Int,
    inputs: StaticTuple[NDBuffer[type, rank], *_],
    context: MojoCallContextPtr = MojoCallContextPtr(),
) raises:
    constrained[is_valid_target[target](), "not a valid target"]()

    with Trace[TraceLevel.OP, target=target]("concat"):

        @parameter
        if is_cpu[target]():
            var inputVec = InlinedFixedVector[NDBuffer[type, rank]](len(inputs))

            @parameter
            for i in range(inputs.size):
                inputVec.append(inputs[i])

            # Dynamic input length is required by `mo.concat_from_list`
            # TODO: Should we just provide a separate implementation for
            # `concat_from_list`, since dynamic input size does not work with
            # static sized input lambda tuple.
            _concat_cpu[
                rank, type, epilogue_fn, single_thread_blocking_override
            ](output, axis, inputVec)
        else:
            _concat_gpu[rank, type, epilogue_fn](
                output,
                axis,
                inputs,
                context.get_device_context(),
            )


fn _concat_inner_most_single_dim[
    rank: Int,
    type: DType,
    num_inputs: Int,
    block_size: Int,
    epilogue_fn: OptionalReg[elementwise_epilogue_type],
](
    output: NDBuffer[type, rank],
    inputs: StaticTuple[NDBuffer[type, rank], num_inputs],
):
    var idx = block_idx.x * block_size + thread_idx.x
    var index = _get_start_indices_of_nth_subvolume_uint[1](
        idx, output.dynamic_shape
    )

    if index > output.num_elements():
        return

    @parameter
    for i in range(num_inputs):
        var out_index = index
        out_index[rank - 1] = i

        @parameter
        if epilogue_fn:
            alias func = epilogue_fn.value()
            func[type, rank, 1](out_index.canonicalize(), inputs[i][index])
        else:
            output[out_index] = inputs[i][index]


@always_inline
fn _concat_gpu_elementwise[
    rank: Int,
    type: DType,
    num_inputs: Int,
    epilogue_fn: OptionalReg[elementwise_epilogue_type],
](
    output: NDBuffer[type, rank],
    axis: Int,
    inputs: StaticTuple[NDBuffer[type, rank], num_inputs],
    ctx: DeviceContext,
) raises:
    # Without parameter dispatch there are 2 extra stack allocations in the GPU kernel
    @parameter
    for i in range(rank):
        if i == axis:
            return _concat_gpu_elementwise[axis=i, epilogue_fn=epilogue_fn](
                output, inputs, ctx
            )


@always_inline
fn _concat_gpu_elementwise[
    axis: Int,
    rank: Int,
    type: DType,
    num_inputs: Int,
    epilogue_fn: OptionalReg[elementwise_epilogue_type],
](
    output: NDBuffer[type, rank],
    inputs: StaticTuple[NDBuffer[type, rank], num_inputs],
    ctx: DeviceContext,
) raises:
    @parameter
    @always_inline
    fn per_output_elem[
        simd_width: Int, _rank: Int
    ](out_index: IndexList[_rank]):
        var in_index = out_index
        in_index[axis] = out_index[axis]

        @parameter
        for i in range(num_inputs):
            var input = inputs[i]
            var input_shape = input.get_shape()

            if in_index[axis] < input_shape[axis]:

                @parameter
                if epilogue_fn:
                    alias func = epilogue_fn.value()
                    func[type, _rank, simd_width](
                        out_index,
                        rebind[NDBuffer[type, _rank]](input)[in_index],
                    )
                else:
                    output[rebind[IndexList[rank]](out_index)] = input[
                        rebind[IndexList[rank]](in_index)
                    ]
                return

            in_index[axis] -= input_shape[axis]

    # Can picture output reshaped to 3D: output_reshape = reshape(output, dims=[-1, concat_dim, -1])
    # where concat_dim = inputs[0][axis] + ... + inputs[n-1][axis].
    # Slices of the innermost dim of output_reshape are contiguous in the corresponding input.
    # Because the inner dim is contiguous we will get coalesced memory access
    # using the elementwise generator with simd_width=1.
    elementwise[per_output_elem, 1, target="gpu"](output.get_shape(), ctx)


@always_inline
fn _concat_gpu[
    rank: Int,
    type: DType,
    epilogue_fn: OptionalReg[elementwise_epilogue_type],
](
    output: NDBuffer[type, rank],
    axis: Int,
    inputs: StaticTuple[NDBuffer[type, rank], *_],
    ctx: DeviceContext,
) raises:
    alias num_inputs = inputs.size
    # Size of outer dims, if 1 we should memcpy to the output buffer.
    var outer_dims = 1
    for i in range(axis):
        # Use input[0], all dims should be equal except axis.
        outer_dims *= inputs[0].dim(i)

    @parameter
    @always_inline
    fn _concat_buffers_contiguously() raises:
        var input_size = 0

        @parameter
        for i in range(num_inputs):
            # Skip empty inputs.
            if inputs[i].num_elements() > 0:
                # TODO: Owning = True or False?
                var outp = DeviceBuffer(
                    ctx,
                    output.data.offset(input_size),
                    inputs[i].num_elements(),
                    owning=False,
                )
                var inp = DeviceBuffer(
                    ctx,
                    inputs[i].data,
                    inputs[i].num_elements(),
                    owning=False,
                )
                ctx.enqueue_copy_device_to_device(
                    outp,
                    inp,
                )

                input_size += inputs[i].num_elements()

    # If outer_dims are ones and it is not a fused kernel, use device-to-device
    # copies.
    @parameter
    if not epilogue_fn:
        if outer_dims == 1:
            return _concat_buffers_contiguously()

    if axis == rank - 1:
        var inner_most_unit_dim = True
        for i in range(num_inputs):
            if inputs[i].dim(axis) != 1:
                inner_most_unit_dim = False
                break

        if inner_most_unit_dim:
            alias block_size = 32
            alias kernel = _concat_inner_most_single_dim[
                rank, type, num_inputs, block_size, epilogue_fn
            ]

            return ctx.enqueue_function[kernel](
                output,
                inputs,
                grid_dim=(inputs[0].num_elements() // block_size),
                block_dim=(block_size),
            )

    _concat_gpu_elementwise[epilogue_fn=epilogue_fn](output, axis, inputs, ctx)


@always_inline
fn _fused_concat_cpu[
    rank: Int,
    type: DType,
    single_thread_blocking_override: Bool,
    input_fn: fn[input_index: Int, width: Int, rank: Int] (
        IndexList[rank]
    ) capturing -> SIMD[type, width],
    output_0_fn: elementwise_epilogue_type,
    size: Int,
](
    axis: Int,
    input_shapes: StaticTuple[IndexList[rank], size],
    output: NDBuffer[type, rank],
    ctx: MojoCallContextPtr,
) raises:
    var offset = 0

    @parameter
    for i in range(input_shapes.size):
        var input_shape = input_shapes[i]

        @parameter
        @always_inline
        fn elementwise_wrapper[
            _width: Int, rank: Int
        ](indices: IndexList[rank]):
            var c = indices
            c[axis] += offset

            # Call the input/output lambda for fused concat kernel.
            output_0_fn[type, rank, width=_width, alignment=1](
                c, input_fn[i, _width, rank](indices)
            )

        # TODO: we can use simd_width > 0 if all inputs are aligned.
        elementwise[
            elementwise_wrapper,
            1,
            use_blocking_impl=single_thread_blocking_override,
        ](input_shape, ctx)
        offset = offset + input_shape[axis]


@always_inline
fn _fused_concat_inner_most_single_dim[
    rank: Int,
    type: DType,
    block_size: Int,
    input_fn: fn[input_index: Int, width: Int, _rank: Int] (
        IndexList[_rank]
    ) capturing -> SIMD[type, width],
    output_0_fn: elementwise_epilogue_type,
    size: Int,
](
    input_shapes: StaticTuple[IndexList[rank], size],
    output: NDBuffer[type, rank],
):
    alias num_inputs = input_shapes.size

    var idx = block_idx.x * block_size + thread_idx.x
    if idx >= product(input_shapes[0], rank):
        return

    var index = _get_start_indices_of_nth_subvolume_uint[1](
        idx, output.dynamic_shape
    )

    @parameter
    for i in range(num_inputs):
        var out_index = index
        out_index[rank - 1] = i

        output_0_fn[type, rank, width=1](
            out_index.canonicalize(),
            input_fn[i, 1, rank](index.canonicalize()),
        )


@always_inline
fn _fused_concat_gpu_elementwise[
    axis: Int,
    rank: Int,
    type: DType,
    input_fn: fn[input_index: Int, width: Int, _rank: Int] (
        IndexList[_rank]
    ) capturing -> SIMD[type, width],
    output_0_fn: elementwise_epilogue_type,
    size: Int,
](
    input_shapes: StaticTuple[IndexList[rank], size],
    output: NDBuffer[type, rank],
    ctx: DeviceContext,
) raises:
    alias num_inputs = input_shapes.size

    @parameter
    @always_inline
    fn per_output_elem[
        simd_width: Int, _rank: Int
    ](out_index: IndexList[_rank]):
        var in_index = out_index
        in_index[axis] = out_index[axis]

        @parameter
        for i in range(num_inputs):
            var input_shape = input_shapes[i]

            if in_index[axis] < input_shape[axis]:
                output_0_fn[type, _rank, width=simd_width, alignment=1](
                    out_index,
                    input_fn[i, simd_width, _rank](in_index),
                )
                return

            in_index[axis] -= input_shape[axis]

    # Can picture output reshaped to 3D: output_reshape = reshape(output, dims=[-1, concat_dim, -1])
    # where concat_dim = inputs[0][axis] + ... + inputs[n-1][axis].
    # Slices of the innermost dim of output_reshape are contiguous in the corresponding input.
    # Because the inner dim is contiguous we will get coalesced memory access
    # using the elementwise generator with simd_width=1.
    elementwise[per_output_elem, 1, target="gpu"](output.get_shape(), ctx)


@always_inline
fn _fused_concat_gpu[
    rank: Int,
    type: DType,
    input_fn: fn[input_index: Int, width: Int, _rank: Int] (
        IndexList[_rank]
    ) capturing -> SIMD[type, width],
    output_0_fn: elementwise_epilogue_type,
    size: Int,
](
    axis: Int,
    input_shapes: StaticTuple[IndexList[rank], size],
    output: NDBuffer[type, rank],
    ctx: DeviceContext,
) raises:
    alias num_inputs = input_shapes.size

    if axis == rank - 1:
        var inner_most_unit_dim = True
        for i in range(num_inputs):
            if (
                input_shapes[i][axis] != 1
                or not input_shapes[i] == input_shapes[0]
            ):
                inner_most_unit_dim = False
                break

        if inner_most_unit_dim:
            alias block_size = 32
            alias kernel = _fused_concat_inner_most_single_dim[
                rank,
                type,
                block_size,
                input_fn,
                output_0_fn,
                size,
            ]

            return ctx.enqueue_function[kernel](
                input_shapes,
                output,
                grid_dim=(
                    ceildiv(
                        product(input_shapes[0], input_shapes[0].size),
                        block_size,
                    )
                ),
                block_dim=(block_size),
            )

    # Without parameter dispatch there are 2 extra stack allocations in the GPU kernel
    @parameter
    for i in range(rank):
        if i == axis:
            return _fused_concat_gpu_elementwise[
                i,
                rank,
                type,
                input_fn,
                output_0_fn,
                size,
            ](input_shapes, output, ctx)


@always_inline
fn fused_concat[
    type: DType,
    rank: Int,
    single_thread_blocking_override: Bool,
    input_fn: fn[input_index: Int, width: Int, _rank: Int] (
        IndexList[_rank]
    ) capturing -> SIMD[type, width],
    output_0_fn: elementwise_epilogue_type,
    target: StringLiteral = "cpu",
](
    axis: Int,
    input_shapes: StaticTuple[IndexList[rank], _],
    output: NDBuffer[type, rank],
    ctx: MojoCallContextPtr,
) raises:
    constrained[is_valid_target[target](), "not a valid target"]()

    with Trace[TraceLevel.OP, target=target]("concat"):

        @parameter
        if is_cpu[target]():
            return _fused_concat_cpu[
                rank,
                type,
                single_thread_blocking_override,
                input_fn,
                output_0_fn,
            ](axis, input_shapes, output, ctx)
        else:
            return _fused_concat_gpu[rank, type, input_fn, output_0_fn](
                axis, input_shapes, output, ctx.get_device_context()
            )
