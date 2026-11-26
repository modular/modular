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
"""Implements `repeat_interleave`, which repeats each tensor element a specified number of times along an axis."""

from std.sys import align_of, simd_width_of

from max.algorithm.functional import elementwise
from max.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    block_idx,
    thread_idx,
)
from max.gpu.host import DeviceContext
from max.gpu.host.info import is_cpu
from max.gpu.primitives import block
from max.gpu.primitives.grid_controls import (
    PDL,
    PDLLevel,
    pdl_launch_attributes,
)
from layout import (
    Coord,
    TensorEngine,
    TensorLayout,
    TileTensor,
    row_major,
)
from layout.coord import DynamicCoord

from std.utils import StaticTuple


def _collapse_dims_around_axis(
    shape: Coord, axis: Int
) raises -> Tuple[Int, Int, Int]:
    comptime rank = type_of(shape).rank
    if axis < 0 or axis >= rank:
        raise Error("axis outside of provided shape")

    comptime if rank == 0:
        return (1, 1, 1)

    var before = 1
    var split = 1
    var after = 1
    comptime for i in range(rank):
        var dim = Int(shape[i])
        if i < axis:
            before *= dim
        elif i == axis:
            split = dim
        else:
            after *= dim

    return (before, split, after)


@inline(.always)
def _repeat_interleave_cpu[
    dtype: DType,
    type_repeats: DType,
](
    input: TileTensor[dtype, ...],
    repeats: TileTensor[type_repeats, ...],
    axis: Int,
    output: TileTensor[mut=True, dtype, ...],
    ctx: DeviceContext,
) raises:
    """
    Fill `output` by repeating values from `input` along `axis` based on the
    values in `repeats` buffer.

    This is intended to implement the same functionality as torch.repeat:
    https://pytorch.org/docs/stable/generated/torch.repeat_interleave.html

    Args:
        input: The input buffer.
        repeats: The number of repetitions each element in input.
        axis: The axis along which to repeat values.
        output: The output buffer.
        ctx: Device context.
    """
    # comptime assert (is_row_major[input.rank](input.layout)) and (
    #     is_row_major[output.rank](output.layout)
    # )
    comptime assert input.flat_rank == output.flat_rank
    comptime assert repeats.flat_rank == 1

    # Compute the shape of the input and result buffers.
    # These are the shapes of the buffers we will be working on.
    var collapsed_input_shape = _collapse_dims_around_axis(
        input.layout.shape_coord(), axis
    )
    var collapsed_output_shape = _collapse_dims_around_axis(
        output.layout.shape_coord(), axis
    )

    assert collapsed_output_shape[0] == collapsed_input_shape[0]
    assert collapsed_output_shape[2] == collapsed_input_shape[2]

    var collapsed_input = input.reshape(row_major(Coord(collapsed_input_shape)))

    var collapsed_output = output.reshape(
        row_major(Coord(collapsed_output_shape))
    )

    var input_repeat_dim = collapsed_input_shape[1]
    var output_repeat_dim = collapsed_output_shape[1]
    var repeat_stride = Int(repeats.dim[0]() > 1)

    # Mapping from offsets in the input tensor to offsets in the output tensor
    # along the repeat axis.
    var offset_mapping = List[Int](unsafe_uninit_length=output_repeat_dim)

    var repeat_offset = 0
    var output_offset = 0
    for input_offset in range(input_repeat_dim):
        var repeat_val = Int(repeats[repeat_offset])
        if repeat_val < 0:
            raise Error("all repeat values must be non-negative")

        for _ in range(repeat_val):
            offset_mapping[output_offset] = input_offset
            output_offset += 1

        repeat_offset += repeat_stride

    # `offset_mapping` is a `List` (not register-passable), so it cannot be
    # captured directly by the unified closure. Capture a `Span` view instead;
    # the `Span` carries the list's origin, keeping it alive through the
    # closure's last use. Everything else is captured by `mut`, matching the
    # original implicit-capture behavior.
    var offset_mapping_view = Span(offset_mapping)

    @inline(.always)
    def func[
        width: Int, alignment: Int = 1
    ](idx: Coord) {var offset_mapping_view, mut}:
        var input_index = Coord(
            Int(idx[0]), offset_mapping_view[Int(idx[1])], Int(idx[2])
        )

        var input_value = collapsed_input.load[
            width=width, alignment=align_of[dtype]()
        ](input_index)
        collapsed_output.store[alignment=align_of[dtype]()](idx, input_value)

    elementwise[simd_width_of[output.dtype]()](
        func, collapsed_output.layout.shape_coord(), ctx
    )


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(BLOCK_SIZE))
)
@__name(t"repeat_interleave_offset_map_{BLOCK_SIZE}")
def _repeat_offset_map_kernel[
    type_repeats: DType,
    BLOCK_SIZE: Int,
    RepeatsLayout: TensorLayout,
    MapLayout: TensorLayout,
    RepeatsEngine: TensorEngine,
    MapEngine: TensorEngine,
](
    repeats: TileTensor[
        type_repeats, RepeatsLayout, ImmutAnyOrigin, Engine=RepeatsEngine
    ],
    offset_map: TileTensor[
        mut=True, DType.int32, MapLayout, MutAnyOrigin, Engine=MapEngine
    ],
    axis_size_arg: Int64,
):
    """Builds the output->input offset map for per-element repeats.

    One block walks the whole axis in BLOCK_SIZE-element chunks, keeping
    a running exclusive-prefix base across chunks: each element's
    repeats value gives its [start, start + repeat) span of the output
    axis, which the element scatter-fills with its own index. The
    caller validates the map length equals the output axis size, so
    every span is in bounds. A block-wide sum finishes each chunk so
    the base advances correctly.
    """
    var axis_size = Int(axis_size_arg)
    var tid = thread_idx.x
    var base: Int64 = 0

    with PDL():
        for chunk in range(0, axis_size, BLOCK_SIZE):
            var selected: Int64 = 0
            var i = chunk + tid
            if i < axis_size:
                var r = Int64(repeats[i])
                if r < 0:
                    r = 0
                selected = r

            # Exclusive prefix within the chunk, plus the running base.
            var start = base + Int64(
                block.prefix_sum[block_size=BLOCK_SIZE, exclusive=True](
                    selected
                )
            )

            if i < axis_size:
                var r = Int(repeats[i])
                if r < 0:
                    r = 0
                var s = Int(start)
                for _ in range(r):
                    offset_map[s] = Int32(i)
                    s += 1

            # Advance the running base by this chunk's total.
            base += Int64(block.sum[block_size=BLOCK_SIZE](selected))


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(BLOCK_SIZE))
)
@__name(t"repeat_interleave_emit_{dtype}_{BLOCK_SIZE}")
def _repeat_interleave_emit_kernel[
    dtype: DType,
    BLOCK_SIZE: Int,
    InputLayout: TensorLayout,
    OutputLayout: TensorLayout,
    MapLayout: TensorLayout,
    InputEngine: TensorEngine,
    OutputEngine: TensorEngine,
    MapEngine: TensorEngine,
](
    input_buffer: TileTensor[
        dtype, InputLayout, ImmutAnyOrigin, Engine=InputEngine
    ],
    output_buffer: TileTensor[
        mut=True, dtype, OutputLayout, MutAnyOrigin, Engine=OutputEngine
    ],
    offset_map: TileTensor[
        DType.int32, MapLayout, ImmutAnyOrigin, Engine=MapEngine
    ],
    in_axis_arg: Int64,
    out_axis_arg: Int64,
    outer_arg: Int64,
    inner_arg: Int64,
):
    """Gathers the per-element-repeat output from the input.

    The collapsed [outer, axis, inner] views are contiguous row-major
    over the original memory, so a flat output index decodes to
    [outer, axis, inner] with two divisions. The source element along
    the axis comes from the device-built offset map; the outer and
    inner coordinates are unchanged. One block covers a BLOCK_SIZE
    chunk of the flat output.
    """
    var in_axis = Int(in_axis_arg)
    var out_axis = Int(out_axis_arg)
    var outer = Int(outer_arg)
    var inner = Int(inner_arg)
    var tid = thread_idx.x
    var flat = block_idx.x * BLOCK_SIZE + tid
    if flat >= outer * out_axis * inner:
        return
    var inner_idx = flat % inner
    var axis_idx = (flat // inner) % out_axis
    var outer_idx = flat // (inner * out_axis)
    with PDL():
        var src_axis = Int(offset_map[axis_idx])
        var src_flat = (
            outer_idx * in_axis * inner + src_axis * inner + inner_idx
        )
        var input_flat = input_buffer.reshape(
            row_major(outer * in_axis * inner)
        )
        var output_flat = output_buffer.reshape(
            row_major(outer * out_axis * inner)
        )
        output_flat.store((flat,), input_flat.load[width=1]((src_flat,)))


def _repeat_interleave_gpu[
    dtype: DType,
    type_repeats: DType,
](
    input: TileTensor[dtype, ...],
    repeats: TileTensor[type_repeats, ...],
    axis: Int,
    output: TileTensor[mut=True, dtype, ...],
    ctx: DeviceContext,
) raises:
    """Fills `output` on a GPU for scalar and per-element repeats.

    Scalar repeats (a size-1 buffer) decode arithmetically: every
    input element expands into `repeat_count` consecutive output
    slots, so the elementwise body divides instead of gathering.
    Per-element repeats build the output->input offset map on the
    device first (a chunked block-wide exclusive scan over the
    repeats, then per-element scatter fills), so no repeats value is
    ever read on the host. Both paths use the collapsed
    [outer, axis, inner] views over the same memory as the CPU path,
    with the extents read on the host and captured by value (the
    device-side layout round-trips misplace indices).
    """
    comptime assert input.flat_rank == output.flat_rank
    comptime assert repeats.flat_rank == 1

    var collapsed_input_shape = _collapse_dims_around_axis(
        input.layout.shape_coord(), axis
    )
    var collapsed_output_shape = _collapse_dims_around_axis(
        output.layout.shape_coord(), axis
    )
    var in_axis = collapsed_input_shape[1]
    var out_axis = collapsed_output_shape[1]

    assert collapsed_output_shape[0] == collapsed_input_shape[0]
    assert collapsed_output_shape[2] == collapsed_input_shape[2]

    var collapsed_input = input.reshape(row_major(Coord(collapsed_input_shape)))
    var collapsed_output = output.reshape(
        row_major(Coord(collapsed_output_shape))
    )

    var num_elements = output.num_elements()
    if num_elements == 0:
        return

    if repeats.dim[0]() == 1:
        # Derive the repeat count from the shapes instead of reading the
        # device-resident repeats buffer on the host (a raw host read of
        # device memory crashes). The shape function already guaranteed
        # out_axis == in_axis * repeats, so the division is exact.
        if in_axis == 0:
            raise Error("repeat_interleave: input axis size must be positive")
        if out_axis % in_axis != 0:
            raise Error(
                "repeat_interleave: output axis size must be a multiple of"
                " the input axis size for scalar repeats"
            )
        var repeat_count = out_axis // in_axis
        if repeat_count < 0:
            raise Error("all repeat values must be non-negative")

        @inline(.always)
        def scalar_func[width: Int, alignment: Int = 1](idx: Coord) {var}:
            var input_index = Coord(
                Int(idx[0]), Int(idx[1]) // repeat_count, Int(idx[2])
            )

            var input_value = collapsed_input.load[
                width=width, alignment=align_of[dtype]()
            ](input_index)
            collapsed_output.store[alignment=align_of[dtype]()](
                idx, input_value
            )

        elementwise[simd_width_of[output.dtype]()](
            scalar_func, collapsed_output.layout.shape_coord(), ctx
        )
        return

    # Per-element repeats: repeats must match the axis size exactly.
    if Int(repeats.dim[0]()) != in_axis:
        raise Error(
            "repeat_interleave: repeats must be size 1 or equal the size of"
            " input[axis]"
        )

    # Build the output->input map on the device. One block walks the
    # axis chunked; the running exclusive prefix keeps starts correct.
    comptime map_block_size = 256
    var map_buffer = ctx.enqueue_create_buffer[.int32](out_axis)
    var offset_map = TileTensor(map_buffer, row_major(out_axis))
    var repeats_imm = repeats.as_imm().as_unsafe_any_origin()
    var map_mut = offset_map.as_unsafe_any_origin()
    comptime map_kernel = _repeat_offset_map_kernel[
        type_repeats,
        map_block_size,
        repeats_imm.LayoutType,
        map_mut.LayoutType,
        repeats_imm.Engine,
        map_mut.Engine,
    ]
    ctx.enqueue_function[map_kernel](
        repeats_imm,
        map_mut,
        Int64(in_axis),
        grid_dim=1,
        block_dim=map_block_size,
        attributes=pdl_launch_attributes(PDLLevel.ON),
    )

    var map_imm = offset_map.as_imm().as_unsafe_any_origin()

    # Gather the output on the device. The collapsed views are
    # contiguous row-major over the original memory, so the emit
    # kernel decodes a flat output index to [outer, axis, inner] and
    # reads the source element through the offset map.
    comptime emit_block_size = 256
    var input_imm = input.as_imm().as_unsafe_any_origin()
    var output_mut = output.as_unsafe_any_origin()
    comptime emit_kernel = _repeat_interleave_emit_kernel[
        dtype,
        emit_block_size,
        input_imm.LayoutType,
        output_mut.LayoutType,
        map_imm.LayoutType,
        input_imm.Engine,
        output_mut.Engine,
        map_imm.Engine,
    ]
    var num_emit_blocks = (
        collapsed_input_shape[0] * out_axis * collapsed_input_shape[2]
        + emit_block_size
        - 1
    ) // emit_block_size
    ctx.enqueue_function[emit_kernel](
        input_imm,
        output_mut,
        map_imm,
        Int64(in_axis),
        Int64(out_axis),
        Int64(collapsed_input_shape[0]),
        Int64(collapsed_input_shape[2]),
        grid_dim=num_emit_blocks,
        block_dim=emit_block_size,
        attributes=pdl_launch_attributes(PDLLevel.ON),
    )


@inline(.always)
def repeat_interleave[
    dtype: DType,
    type_repeats: DType,
    //,
](
    input: TileTensor[dtype, ...],
    repeats: TileTensor[type_repeats, ...],
    axis: Int,
    output: TileTensor[mut=True, dtype, ...],
    ctx: DeviceContext,
) raises:
    """Fills `output` by repeating `input` along `axis`, on `target`.

    Both scalar and per-element repeats run on CPU and GPU (see
    `_repeat_interleave_gpu` for the GPU decomposition).
    """

    comptime if ctx.target.is_cpu():
        _repeat_interleave_cpu(input, repeats, axis, output, ctx)
    else:
        _repeat_interleave_gpu[dtype, type_repeats](
            input, repeats, axis, output, ctx
        )


@inline(.always)
def repeat_interleave_shape[
    type_repeats: DType,
](
    input_shape: Coord,
    repeats: TileTensor[type_repeats, ...],
    axis: Int,
) raises -> DynamicCoord[.int, type_of(input_shape).rank]:
    """
    Computes the output shape of `repeat_interleave` for an input of shape
    `input_shape`, `repeats`, and `axis`.

    The returned shape is `input_shape` with the size along `axis`
    replaced by the summed `repeats` values, or by `input_shape[axis]`
    multiplied by the single repeat value when `repeats` is a size-1
    broadcast.

    Parameters:
        type_repeats: The integral element type of `repeats`.

    Args:
        input_shape: The shape of the input tensor.
        repeats: A one-dimensional integral tensor of per-element repeat
            counts, either size 1 or equal to `input_shape[axis]`.
        axis: The axis along which elements are repeated.

    Returns:
        `input_shape` with the `axis` dimension updated to the total repeated
        size.

    Raises:
        Error: If `axis` is outside `[0, rank)`, or `repeats` is neither size 1
            nor the size of `input_shape[axis]`.
    """
    comptime assert type_repeats.is_integral()
    comptime assert repeats.flat_rank == 1

    comptime rank = type_of(input_shape).rank
    if axis < 0 or axis >= rank:
        raise Error("repeat_interleave: axis out of range")
    var axis_dim = 0
    comptime for i in range(rank):
        if i == axis:
            axis_dim = Int(input_shape[i])

    var repeats_size = repeats.dim[0]()
    if repeats_size != 1 and repeats_size != Scalar[repeats.linear_idx_type](
        axis_dim
    ):
        raise Error(
            "repeat_interleave: repeats must be size 1 or equal to "
            "the size of input[axis]"
        )

    var total_repeats = 0
    comptime assert repeats_size.dtype.is_integral()
    for i in range(repeats_size):
        total_repeats += Int(repeats[i])

    # If the repeats is size 1, the repeat is treated as a broadcast
    var new_axis_dim = (
        axis_dim * total_repeats if repeats_size == 1 else total_repeats
    )
    var result = DynamicCoord[.int, rank]()
    comptime for i in range(rank):
        result.replace_dynamic[i](
            new_axis_dim if i == axis else Int(input_shape[i])
        )
    return result
