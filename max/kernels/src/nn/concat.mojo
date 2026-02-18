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

from collections import Optional
from math import align_down, align_up, ceildiv

from sys._build import is_debug_build
from sys.info import simd_width_of, size_of

from algorithm.functional import (
    _get_start_indices_of_nth_subvolume,
    _get_start_indices_of_nth_subvolume_uint,
    elementwise,
    sync_parallelize,
)
from gpu import block_idx, thread_idx
from gpu.host import DeviceBuffer, DeviceContext
from gpu.host.info import is_cpu, is_valid_target
from layout._coord import Coord, Idx, coord_to_index_list
from layout._layout import TensorLayout, row_major
from layout._tile_tensor import TileTensor
from memory import memcpy
from runtime.asyncrt import DeviceContextPtr
from runtime.tracing import Trace, TraceLevel, get_safe_task_id

from utils import IndexList, StaticTuple, product

from .gather_scatter import normalize_neg_index

comptime elementwise_epilogue_type = fn[
    c_type: DType, rank: Int, width: Int = 1, *, alignment: Int = 1
](IndexList[rank], SIMD[c_type, width]) capturing -> None


# ===-----------------------------------------------------------------------===#
# concat
# ===-----------------------------------------------------------------------===#


@always_inline
fn memcpy_or_fuse[
    rank: Int,
    dtype: DType,
    epilogue_fn: Optional[elementwise_epilogue_type],
](
    dest_data: UnsafePointer[mut=True, Int8],
    out_byte_offset: Int,
    src_data: UnsafePointer[Int8],
    n: Int,
    out_shape: IndexList[rank, ...],
) raises:
    @parameter
    if not epilogue_fn:
        memcpy(dest=dest_data + out_byte_offset, src=src_data, count=n)
    else:
        comptime func = epilogue_fn.value()
        comptime simd_width = simd_width_of[dtype]()

        var typed_offset = out_byte_offset // size_of[dtype]()
        var typed_len = n // size_of[dtype]()
        debug_assert(
            n % size_of[dtype]() == 0
            and out_byte_offset % size_of[dtype]() == 0,
            "offset and length must be dividable by size_of[dtype]",
        )

        # Cast
        var shape_1d = IndexList[1](typed_len)
        var typed_src = src_data.bitcast[Scalar[dtype]]()
        var input = TileTensor(
            typed_src,
            row_major(Coord(shape_1d)),
        )

        @parameter
        @always_inline
        fn epilogue_wrapper[
            simd_width: Int, _rank: Int, alignment: Int = 1
        ](index: IndexList[_rank]):
            var coord = Coord(index)
            comptime assert coord.flat_rank == input.flat_rank
            var load = input.load[width=simd_width](coord)

            # Convert the linearized address back to the n-D indices.
            comptime assert _rank == 1
            var out_index = _get_start_indices_of_nth_subvolume[0](
                index[0] + typed_offset,
                out_shape,
            )

            func[dtype, rank, simd_width](
                out_index.cast[DType.int64](),
                load,
            )
            return

        # We must run scalar to be conservative. This is because the fused
        # output lambda might operate on views (e.g., broadcast) that does not
        # always work with indices produced from a linearized address.
        elementwise[epilogue_wrapper, simd_width=1](shape_1d)


@fieldwise_init
struct _Span(TrivialRegisterPassable):
    var start: Int
    var end: Int

    @always_inline("nodebug")
    fn empty(self) -> Bool:
        return not (self.start < self.end)

    @always_inline("nodebug")
    fn intersect(self, other: Self) -> Self:
        return Self(max(self.start, other.start), min(self.end, other.end))


@fieldwise_init
struct _CanonicallyReshapedBuffer[mut: Bool, //, origin: Origin[mut=mut]](
    TrivialRegisterPassable
):
    var data: UnsafePointer[Int8, Self.origin]
    var h: Int
    var w: Int
    var c: Int


fn _canonical_reshape[
    dtype: DType
](
    buf: TileTensor[dtype, address_space = AddressSpace.GENERIC, ...],
    axis: Int,
) -> _CanonicallyReshapedBuffer[buf.origin]:
    var shape = coord_to_index_list(buf.layout.shape_coord())
    var h = product(shape, 0, axis)
    var w = Int(buf.dim(axis))
    var c = product(shape, axis + 1, buf.rank) * size_of[dtype]()
    return _CanonicallyReshapedBuffer(buf.ptr.bitcast[Int8](), h, w, c)


fn _canonical_reshape_output[
    input_origin: ImmutOrigin,
    InputLayoutType: TensorLayout,
    //,
    dtype: DType,
](
    out_buf: TileTensor[
        mut=True, dtype, address_space = AddressSpace.GENERIC, ...
    ],
    axis: Int,
    inputs: List[TileTensor[dtype, InputLayoutType, input_origin]],
) -> _CanonicallyReshapedBuffer[out_buf.origin]:
    var input0_canon = _canonical_reshape(inputs[0], axis)
    var out_w = input0_canon.w
    for i in range(1, len(inputs)):
        out_w += Int(inputs[i].dim(axis))
    return _CanonicallyReshapedBuffer(
        out_buf.ptr.bitcast[Int8](),
        input0_canon.h,
        out_w,
        input0_canon.c,
    )


fn _concat_parallel[
    input_origin: ImmutOrigin,
    InputLayoutType: TensorLayout,
    //,
    dtype: DType,
    epilogue_fn: Optional[elementwise_epilogue_type],
](
    output: TileTensor[
        mut=True, dtype, address_space = AddressSpace.GENERIC, ...
    ],
    axis: Int,
    inputs: List[TileTensor[dtype, InputLayoutType, input_origin]],
) raises:
    var output_canon = _canonical_reshape_output(output, axis, inputs)

    var output_h = output_canon.h
    var output_w = output_canon.w
    var output_c = output_canon.c
    var output_wc = output_w * output_c
    var output_data = output_canon.data

    var total_output_bytes = output_h * output_wc

    comptime KB = 1024
    comptime parallel_chunk_size = 64 * KB  # TODO autotune
    var num_chunks = ceildiv(total_output_bytes, parallel_chunk_size)

    @__copy_capture(
        total_output_bytes, output_h, output_c, output_data, output_wc
    )
    @parameter
    fn do_chunk(chunk_index: Int) raises:
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
                    memcpy_or_fuse[output.rank, dtype, epilogue_fn](
                        output_data,
                        output_wc_offset
                        + overlap_rel_start // input_wc * output_wc
                        + overlap_rel_start % input_wc,
                        input_data + overlap_rel_start,
                        overlap_rel_end - overlap_rel_start,
                        rebind[IndexList[output.rank]](
                            coord_to_index_list(output.layout.shape_coord())
                        ),
                    )
                else:
                    # OK, we have maybe stragglers on the start and end, and a
                    # nice solid middle section -- var's handle those
                    # separately.
                    # First, leading stragglers:
                    memcpy_or_fuse[output.rank, dtype, epilogue_fn](
                        output_data,
                        output_wc_offset
                        + overlap_rel_start // input_wc * output_wc
                        + overlap_rel_start % input_wc,
                        input_data + overlap_rel_start,
                        overlap_full_rel_start - overlap_rel_start,
                        rebind[IndexList[output.rank]](
                            coord_to_index_list(output.layout.shape_coord())
                        ),
                    )
                    # Now, fully-aligned sections:
                    var in_ptr = input_data + overlap_full_rel_start
                    var end_in_ptr = input_data + overlap_full_rel_end
                    var out_ptr_offset = (
                        output_wc_offset
                        + overlap_full_rel_start // input_wc * output_wc
                    )

                    while in_ptr < end_in_ptr:
                        memcpy_or_fuse[output.rank, dtype, epilogue_fn](
                            output_data,
                            out_ptr_offset,
                            in_ptr,
                            input_wc,
                            rebind[IndexList[output.rank]](
                                coord_to_index_list(output.layout.shape_coord())
                            ),
                        )
                        in_ptr += input_wc
                        out_ptr_offset += output_wc
                    # Lastly, trailing stragglers:
                    memcpy_or_fuse[output.rank, dtype, epilogue_fn](
                        output_data,
                        out_ptr_offset,
                        in_ptr,
                        overlap_rel_end - overlap_full_rel_end,
                        rebind[IndexList[output.rank]](
                            coord_to_index_list(output.layout.shape_coord())
                        ),
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
    input_origin: ImmutOrigin,
    InputLayoutType: TensorLayout,
    //,
    dtype: DType,
    epilogue_fn: Optional[elementwise_epilogue_type],
](
    output: TileTensor[
        mut=True, dtype, address_space = AddressSpace.GENERIC, ...
    ],
    axis: Int,
    inputs: List[TileTensor[dtype, InputLayoutType, input_origin]],
) raises:
    """Concatenate inputs along axis and store in output.

    This simplifies the implementation by reshaping the output and inputs into 3D
    buffers. input i has dims [h, wi, c]. The output has dims [h, sum(wi), c] where
    i ranges from [0, num_inputs).

    Reshaping the buffer does not change the memory layout. After reshaping to 3D
    it is easy to visualize that the inputs can be copied in w x c sized
    contiguous slices along the h dimension.

    """

    var h = product(
        coord_to_index_list(inputs[0].layout.shape_coord()), 0, axis
    )
    var c = product(
        coord_to_index_list(inputs[0].layout.shape_coord()),
        axis + 1,
        output.rank,
    )

    var w_out: Int = 0
    for i in range(len(inputs)):
        w_out += Int(inputs[i].dim(axis))

    var stride_h_out = w_out * c
    var stride_w_out = c

    var w_offset: Int = 0
    for i in range(len(inputs)):
        # copy one w x c slice along h at a time
        var w = Int(inputs[i].dim(axis))
        for j in range(h):
            var input_offset = j * w * c
            var output_offset = j * stride_h_out + w_offset * stride_w_out
            # these slices are contiguous
            memcpy_or_fuse[output.rank, dtype, epilogue_fn](
                output.ptr.bitcast[Int8](),
                output_offset * size_of[dtype](),
                (inputs[i].ptr + input_offset).bitcast[Int8](),
                w * c * size_of[dtype](),
                rebind[IndexList[output.rank]](
                    coord_to_index_list(output.layout.shape_coord())
                ),
            )
        w_offset += w


@always_inline
fn _concat_inner[
    input_origin: ImmutOrigin,
    InputLayoutType: TensorLayout,
    //,
    dtype: DType,
    epilogue_fn: Optional[elementwise_epilogue_type],
](
    output: TileTensor[
        mut=True, dtype, address_space = AddressSpace.GENERIC, ...
    ],
    inputs: List[TileTensor[dtype, InputLayoutType, input_origin]],
) raises:
    var num_elems_copied: Int = 0
    for i in range(len(inputs)):
        var buffer_len = inputs[i].numel()
        memcpy_or_fuse[output.rank, dtype, epilogue_fn](
            output.ptr.bitcast[Int8](),
            num_elems_copied * size_of[dtype](),
            inputs[i].ptr.bitcast[Int8](),
            buffer_len * size_of[dtype](),
            rebind[IndexList[output.rank]](
                coord_to_index_list(output.layout.shape_coord())
            ),
        )
        num_elems_copied += buffer_len


@always_inline
fn _check_input_consistency[
    input_origin: ImmutOrigin,
    InputLayoutType: TensorLayout,
    //,
    dtype: DType,
](axis: Int, inputs: List[TileTensor[dtype, InputLayoutType, input_origin]],):
    @parameter
    if not is_debug_build():
        return
    # check inputs have same rank and same dims except for axis dim
    for i in range(len(inputs)):
        for j in range(inputs[i].rank):
            debug_assert(
                j == axis or inputs[0].dim(j) == inputs[i].dim(j),
                (
                    "all concat inputs must have the same dimensions in the"
                    " non-concat axes"
                ),
            )


@always_inline
fn _concat_serial[
    input_origin: ImmutOrigin,
    InputLayoutType: TensorLayout,
    //,
    dtype: DType,
    epilogue_fn: Optional[elementwise_epilogue_type],
](
    output: TileTensor[
        mut=True, dtype, address_space = AddressSpace.GENERIC, ...
    ],
    axis: Int,
    inputs: List[TileTensor[dtype, InputLayoutType, input_origin]],
) raises:
    _check_input_consistency[dtype](axis, inputs)

    var all_outer_dims_singvaron = True
    for i in range(axis):
        if inputs[0].dim(i) == 1:
            continue

        all_outer_dims_singvaron = False
        break

    if all_outer_dims_singvaron:
        _concat_inner[dtype, epilogue_fn](output, inputs)
        return

    _concat[dtype, epilogue_fn](output, axis, inputs)


@always_inline
fn _concat_small[
    input_origin: ImmutOrigin,
    InputLayoutType: TensorLayout,
    //,
    dtype: DType,
    epilogue_fn: Optional[elementwise_epilogue_type],
](
    output: TileTensor[
        mut=True, dtype, address_space = AddressSpace.GENERIC, ...
    ],
    axis: Int,
    inputs: List[TileTensor[dtype, InputLayoutType, input_origin]],
) raises:
    comptime single_thread_blocking_override = True
    comptime simd_width = simd_width_of[dtype]()

    @parameter
    @always_inline
    fn concat_lambda[
        simd_width: Int, rank: Int, alignment: Int = 1
    ](out_index: IndexList[rank]):
        # Concatenating [:, 10, :], [:, 20, :], [:, 30, :] results in shape
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
            if target_dim < Int(input.dim(axis)):
                var in_index = out_index
                in_index[axis] = target_dim
                var coord = Coord(in_index)
                comptime assert coord.flat_rank == input.flat_rank
                var load = input.load[width=simd_width](coord)

                @parameter
                if epilogue_fn:
                    comptime func = epilogue_fn.value()
                    func[dtype, rank, simd_width](out_index, load)
                else:
                    var coord = Coord(out_index)
                    comptime assert coord.flat_rank == output.flat_rank
                    output.store[width=simd_width](coord, load)
                return
            else:
                # Keep looking...
                target_dim -= Int(input.dim(axis))

    # We need to check it's safe to simd_load from each input.
    var inputs_simd_aligned = True
    for i in range(len(inputs)):
        if (
            inputs[i].dim(output.rank - 1)
            % Scalar[inputs.T.linear_idx_type](simd_width)
            != 0
        ):
            inputs_simd_aligned = False

    # If we are concat'ing along the last dimension we can do a simd load.
    if axis == output.rank - 1 and inputs_simd_aligned:
        elementwise[
            concat_lambda,
            simd_width=simd_width,
            use_blocking_impl=single_thread_blocking_override,
        ](coord_to_index_list(output.layout.shape_coord()))
    else:
        # Otherwise we must run scalar.
        elementwise[
            concat_lambda,
            simd_width=1,
            use_blocking_impl=single_thread_blocking_override,
        ](coord_to_index_list(output.layout.shape_coord()))


@always_inline
fn _concat_cpu[
    input_origin: ImmutOrigin,
    InputLayoutType: TensorLayout,
    //,
    dtype: DType,
    epilogue_fn: Optional[elementwise_epilogue_type],
    single_thread_blocking_override: Bool,
](
    output: TileTensor[
        mut=True, dtype, address_space = AddressSpace.GENERIC, ...
    ],
    axis: Int,
    inputs: List[TileTensor[dtype, InputLayoutType, input_origin]],
) raises:
    @parameter
    if single_thread_blocking_override:
        return _concat_small[dtype, epilogue_fn](output, axis, inputs)

    _check_input_consistency[dtype](axis, inputs)

    @always_inline
    @parameter
    fn dispatch_serial(unused_thread_idx: Int) raises:
        _concat_serial[dtype, epilogue_fn](output, axis, inputs)

    comptime KB = 1024
    comptime min_work_for_parallel = 128 * KB  # TODO: autotune

    var output_bytes = output.numel() * size_of[dtype]()

    if output_bytes < min_work_for_parallel:
        # The dispatch_serial closure captures the stack allocated
        # Buffer, so this kernel must be run synchronously.
        sync_parallelize[dispatch_serial](1)
    else:
        _concat_parallel[epilogue_fn=epilogue_fn](output, axis, inputs)


@always_inline
fn concat_shape[
    input_origin: ImmutOrigin,
    InputLayoutType: TensorLayout,
    //,
    input_type: DType,
    single_thread_blocking_override: Bool,
](
    input_bufs: List[TileTensor[input_type, InputLayoutType, input_origin]],
    axis: Int,
) raises -> IndexList[InputLayoutType.rank]:
    """
    Compute the output shape of a `pad` operation, and assert the inputs are
    compatible.

    Parameters:
        input_origin: Origin of the input tensor.
        InputLayoutType: Layout type of the input tensor.
        input_type: Type of the input tensor.
        single_thread_blocking_override: If True, then the operation is run
          synchronously using a single thread.

    Args:
        input_bufs: The input tensors list.
        axis: The axis.

    Returns:
        The output shape.
    """

    # extract hyper parameters
    var normalized_axis = normalize_neg_index(axis, InputLayoutType.rank)

    @parameter
    @always_inline
    fn shape_equal_ignore_axis(
        s1: IndexList[InputLayoutType.rank],
        s2: IndexList[InputLayoutType.rank],
    ) -> Bool:
        for i in range(InputLayoutType.rank):
            if i != axis and s1[i] != s2[i]:
                return False
        return True

    var concat_axis_dim_sum = 0
    for i in range(len(input_bufs)):
        concat_axis_dim_sum += Int(input_bufs[i].dim(normalized_axis))
        if not shape_equal_ignore_axis(
            rebind[IndexList[InputLayoutType.rank]](
                coord_to_index_list(input_bufs[0].layout.shape_coord())
            ),
            rebind[IndexList[InputLayoutType.rank]](
                coord_to_index_list(input_bufs[i].layout.shape_coord())
            ),
        ):
            raise Error(
                "[concat_from_list] input shapes must match except at concat"
                " axis"
            )

    # compute and return the output shape
    var output_shape = rebind[IndexList[InputLayoutType.rank]](
        coord_to_index_list(input_bufs[0].layout.shape_coord())
    )
    output_shape[normalized_axis] = concat_axis_dim_sum
    return output_shape


@always_inline
fn concat[
    input_origin: ImmutOrigin,
    InputLayoutType: TensorLayout,
    //,
    dtype: DType,
    single_thread_blocking_override: Bool,
    target: StaticString = "cpu",
    epilogue_fn: Optional[elementwise_epilogue_type] = None,
](
    output: TileTensor[
        mut=True, dtype, address_space = AddressSpace.GENERIC, ...
    ],
    axis: Int,
    inputs: StaticTuple[
        TileTensor[dtype, InputLayoutType, input_origin],
        ...,
    ],
    context: DeviceContextPtr = DeviceContextPtr(),
) raises:
    comptime assert is_valid_target[target](), "not a valid target"

    with Trace[TraceLevel.OP, target=target](
        "concat", task_id=get_safe_task_id(context)
    ):

        @parameter
        if is_cpu[target]():
            var inputVec = List[
                TileTensor[dtype, InputLayoutType, input_origin]
            ](capacity=len(inputs))

            for i in range(inputs.size):
                inputVec.append(inputs[i])

            # Dynamic input length is required by `mo.concat_from_list`
            # TODO: Should we just provide a separate implementation for
            # `concat_from_list`, since dynamic input size does not work with
            # static sized input lambda tuple.
            _concat_cpu[dtype, epilogue_fn, single_thread_blocking_override](
                output, axis, inputVec
            )
        else:
            _concat_gpu[dtype, epilogue_fn](
                # This is safe since `output` being an arg will keep the origin alive
                # for the duration of this call.
                output,
                axis,
                inputs,
                context.get_device_context(),
            )


fn _concat_inner_most_single_dim[
    OutputLayoutType: TensorLayout,
    output_origin: MutOrigin,
    InputLayoutType: TensorLayout,
    input_origin: ImmutOrigin,
    //,
    dtype: DType,
    num_inputs: Int,
    block_size: Int,
    epilogue_fn: Optional[elementwise_epilogue_type],
](
    output: TileTensor[dtype, OutputLayoutType, output_origin],
    inputs: StaticTuple[
        TileTensor[dtype, InputLayoutType, input_origin],
        num_inputs,
    ],
):
    var idx = block_idx.x * UInt(block_size) + thread_idx.x
    if idx >= UInt(output.numel()):
        return

    var index = _get_start_indices_of_nth_subvolume_uint[1](
        idx, coord_to_index_list(output.layout.shape_coord())
    )
    var in_coord = Coord(index)
    comptime assert in_coord.flat_rank == inputs.element_type.flat_rank

    @parameter
    for i in range(num_inputs):
        var out_index = rebind[IndexList[output.rank]](index.canonicalize())
        out_index[output.rank - 1] = i
        var out_coord = Coord(out_index)
        comptime assert out_coord.flat_rank == output.flat_rank

        @parameter
        if epilogue_fn:
            comptime func = epilogue_fn.value()
            func[dtype, output.rank, 1](
                out_index, inputs[i].load[width=1](in_coord)
            )
        else:
            output.store(out_coord, inputs[i].load[width=1](in_coord))


@always_inline
fn _concat_gpu_elementwise[
    input_origin: ImmutOrigin,
    InputLayoutType: TensorLayout,
    //,
    dtype: DType,
    num_inputs: Int,
    epilogue_fn: Optional[elementwise_epilogue_type],
](
    output: TileTensor[
        mut=True, dtype, address_space = AddressSpace.GENERIC, ...
    ],
    axis: Int,
    inputs: StaticTuple[
        TileTensor[dtype, InputLayoutType, input_origin],
        num_inputs,
    ],
    ctx: DeviceContext,
) raises:
    # Without parameter dispatch there are 2 extra stack allocations in the GPU kernel
    @parameter
    for i in range(output.rank):
        if i == axis:
            return _concat_gpu_elementwise[axis=i, epilogue_fn=epilogue_fn](
                output, inputs, ctx
            )


@always_inline
fn _concat_gpu_elementwise[
    input_origin: ImmutOrigin,
    InputLayoutType: TensorLayout,
    //,
    axis: Int,
    dtype: DType,
    num_inputs: Int,
    epilogue_fn: Optional[elementwise_epilogue_type],
](
    output: TileTensor[
        mut=True, dtype, address_space = AddressSpace.GENERIC, ...
    ],
    inputs: StaticTuple[
        TileTensor[dtype, InputLayoutType, input_origin],
        num_inputs,
    ],
    ctx: DeviceContext,
) raises:
    @parameter
    @always_inline
    fn per_output_elem[
        simd_width: Int, _rank: Int, alignment: Int = 1
    ](out_index: IndexList[_rank]):
        var in_index = out_index
        in_index[axis] = out_index[axis]
        var out_coord = Coord(out_index)
        comptime assert out_coord.flat_rank == output.flat_rank

        @parameter
        for i in range(num_inputs):
            var input = inputs[i]
            var input_shape = coord_to_index_list(input.layout.shape_coord())

            if in_index[axis] < input_shape[axis]:
                var in_coord = Coord(in_index)
                comptime assert in_coord.flat_rank == input.flat_rank

                @parameter
                if epilogue_fn:
                    comptime func = epilogue_fn.value()
                    func[dtype, _rank, simd_width](
                        out_index,
                        input.load[width=1](in_coord),
                    )
                else:
                    output.store(out_coord, input.load[width=1](in_coord))
                return

            in_index[axis] -= input_shape[axis]

    # Can picture output reshaped to 3D: output_reshape = reshape(output, dims=[-1, concat_dim, -1])
    # where concat_dim = inputs[0][axis] + ... + inputs[n-1][axis].
    # Slices of the innermost dim of output_reshape are contiguous in the corresponding input.
    # Because the inner dim is contiguous we will get coalesced memory access
    # using the elementwise generator with simd_width=1.
    elementwise[per_output_elem, 1, target="gpu"](
        coord_to_index_list(output.layout.shape_coord()), ctx
    )


@always_inline
fn _concat_gpu[
    input_origin: ImmutOrigin,
    InputLayoutType: TensorLayout,
    //,
    dtype: DType,
    epilogue_fn: Optional[elementwise_epilogue_type],
](
    output: TileTensor[
        mut=True, dtype, address_space = AddressSpace.GENERIC, ...
    ],
    axis: Int,
    inputs: StaticTuple[
        TileTensor[dtype, InputLayoutType, input_origin],
        ...,
    ],
    ctx: DeviceContext,
) raises:
    comptime num_inputs = inputs.size
    # Size of outer dims, if 1 we should memcpy to the output buffer.
    var outer_dims = 1
    for i in range(axis):
        # Use input[0], all dims should be equal except axis.
        outer_dims *= Int(inputs[0].dim(i))

    @parameter
    @always_inline
    fn _concat_buffers_contiguously() raises:
        var input_size = 0

        @parameter
        for i in range(num_inputs):
            # Skip empty inputs.
            if inputs[i].numel() > 0:
                # TODO: Owning = True or False?
                var outp = DeviceBuffer(
                    ctx,
                    output.ptr + input_size,
                    inputs[i].numel(),
                    owning=False,
                )
                var inp = DeviceBuffer(
                    ctx,
                    inputs[i].ptr,
                    inputs[i].numel(),
                    owning=False,
                )
                ctx.enqueue_copy(
                    outp,
                    inp,
                )

                input_size += inputs[i].numel()

    # If outer_dims are ones and it is not a fused kernel, use device-to-device
    # copies.
    @parameter
    if not epilogue_fn:
        if outer_dims == 1:
            return _concat_buffers_contiguously()

    if axis == output.rank - 1:
        var inner_most_unit_dim = True
        for i in range(num_inputs):
            if inputs[i].dim(axis) != 1:
                inner_most_unit_dim = False
                break

        if inner_most_unit_dim:
            comptime block_size = 32
            comptime kernel = _concat_inner_most_single_dim[
                OutputLayoutType = output.LayoutType,
                output_origin = output.origin,
                InputLayoutType=InputLayoutType,
                input_origin=input_origin,
                dtype,
                num_inputs,
                block_size,
                epilogue_fn,
            ]

            return ctx.enqueue_function[kernel, kernel](
                output,
                inputs,
                grid_dim=(inputs[0].numel() // block_size),
                block_dim=(block_size),
            )

    _concat_gpu_elementwise[epilogue_fn=epilogue_fn](output, axis, inputs, ctx)


@always_inline
fn _fused_concat_cpu[
    rank: Int,
    dtype: DType,
    single_thread_blocking_override: Bool,
    input_fn: fn[input_index: Int, width: Int, rank: Int](
        IndexList[rank]
    ) capturing -> SIMD[dtype, width],
    output_0_fn: elementwise_epilogue_type,
    size: Int,
](
    axis: Int,
    input_shapes: StaticTuple[IndexList[rank], size],
    output: TileTensor[
        mut=True, dtype, address_space = AddressSpace.GENERIC, ...
    ],
    ctx: DeviceContextPtr,
) raises:
    var offset = 0

    @parameter
    for i in range(input_shapes.size):
        var input_shape = input_shapes[i]

        @parameter
        @always_inline
        fn elementwise_wrapper[
            _width: Int, rank: Int, alignment: Int = 1
        ](indices: IndexList[rank]):
            var c = indices
            c[axis] += offset

            # Call the input/output lambda for fused concat kernel.
            output_0_fn[dtype, rank, width=_width, alignment=1](
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
    OutputLayoutType: TensorLayout,
    output_origin: MutOrigin,
    //,
    rank: Int,
    dtype: DType,
    block_size: Int,
    input_fn: fn[input_index: Int, width: Int, _rank: Int](
        IndexList[_rank]
    ) capturing -> SIMD[dtype, width],
    output_0_fn: elementwise_epilogue_type,
    size: Int,
](
    input_shapes: StaticTuple[IndexList[rank], size],
    output: TileTensor[dtype, OutputLayoutType, output_origin],
):
    comptime num_inputs = input_shapes.size

    var idx = block_idx.x * UInt(block_size) + thread_idx.x
    if idx >= UInt(product(input_shapes[0], rank)):
        return

    var index = _get_start_indices_of_nth_subvolume_uint[1](
        idx, coord_to_index_list(output.layout.shape_coord())
    )

    @parameter
    for i in range(num_inputs):
        var out_index = index
        out_index[rank - 1] = i

        output_0_fn[dtype, rank, width=1](
            rebind[IndexList[rank]](out_index.canonicalize()),
            input_fn[i, 1, rank](rebind[IndexList[rank]](index.canonicalize())),
        )


@always_inline
fn _fused_concat_gpu_elementwise[
    axis: Int,
    rank: Int,
    dtype: DType,
    input_fn: fn[input_index: Int, width: Int, _rank: Int](
        IndexList[_rank]
    ) capturing -> SIMD[dtype, width],
    output_0_fn: elementwise_epilogue_type,
    size: Int,
](
    input_shapes: StaticTuple[IndexList[rank], size],
    output: TileTensor[
        mut=True, dtype, address_space = AddressSpace.GENERIC, ...
    ],
    ctx: DeviceContext,
) raises:
    comptime num_inputs = input_shapes.size

    @parameter
    @always_inline
    fn per_output_elem[
        simd_width: Int, _rank: Int, alignment: Int = 1
    ](out_index: IndexList[_rank]):
        var in_index = out_index
        in_index[axis] = out_index[axis]

        @parameter
        for i in range(num_inputs):
            var input_shape = input_shapes[i]

            if in_index[axis] < input_shape[axis]:
                output_0_fn[dtype, _rank, width=simd_width, alignment=1](
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
    elementwise[per_output_elem, 1, target="gpu"](
        coord_to_index_list(output.layout.shape_coord()), ctx
    )


@always_inline
fn _fused_concat_gpu[
    rank: Int,
    dtype: DType,
    input_fn: fn[input_index: Int, width: Int, _rank: Int](
        IndexList[_rank]
    ) capturing -> SIMD[dtype, width],
    output_0_fn: elementwise_epilogue_type,
    size: Int,
](
    axis: Int,
    input_shapes: StaticTuple[IndexList[rank], size],
    output: TileTensor[mut=True, dtype],
    ctx: DeviceContext,
) raises:
    comptime num_inputs = input_shapes.size

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
            comptime block_size = 32
            comptime kernel = _fused_concat_inner_most_single_dim[
                OutputLayoutType = output.LayoutType,
                output_origin = output.origin,
                rank,
                dtype,
                block_size,
                input_fn,
                output_0_fn,
                size,
            ]

            return ctx.enqueue_function[kernel, kernel](
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
                dtype,
                input_fn,
                output_0_fn,
                size,
            ](input_shapes, output, ctx)


@always_inline
fn fused_concat[
    dtype: DType,
    rank: Int,
    single_thread_blocking_override: Bool,
    input_fn: fn[input_index: Int, width: Int, _rank: Int](
        IndexList[_rank]
    ) capturing -> SIMD[dtype, width],
    output_0_fn: elementwise_epilogue_type,
    target: StaticString = "cpu",
](
    axis: Int,
    input_shapes: StaticTuple[IndexList[rank], _],
    output: TileTensor[mut=True, dtype],
    ctx: DeviceContextPtr,
) raises:
    comptime assert is_valid_target[target](), "not a valid target"

    with Trace[TraceLevel.OP, target=target](
        "concat", task_id=get_safe_task_id(ctx)
    ):

        @parameter
        if is_cpu[target]():
            return _fused_concat_cpu[
                rank,
                dtype,
                single_thread_blocking_override,
                input_fn,
                output_0_fn,
            ](axis, input_shapes, output, ctx)
        else:
            return _fused_concat_gpu[rank, dtype, input_fn, output_0_fn](
                axis,
                input_shapes,
                output.as_any_origin(),
                ctx.get_device_context(),
            )
