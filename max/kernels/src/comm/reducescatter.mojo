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
"""Multi-GPU reducescatter implementation for distributed tensor reduction across GPUs.
"""

from collections import InlineArray
from collections.optional import Optional
from builtin.variadics import Variadic

from layout import Coord, TileTensor, Idx
from layout._layout import TensorLayout, Layout, row_major
from layout.coord import _CoordToDynamic
from gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    global_idx,
    grid_dim,
)
from gpu.primitives.grid_controls import (
    PDLLevel,
    launch_dependent_grids,
    pdl_launch_attributes,
    wait_on_dependent_grids,
)
from gpu.host import DeviceContext, get_gpu_target
from gpu.memory import Consistency, ReduceOp, multimem_ld_reduce
from utils import IndexList, StaticTuple
from utils.numerics import get_accum_type

from gpu.intrinsics import (
    Scope,
)
from math import ceildiv
from sys import simd_width_of, align_of, is_amd_gpu

from .sync import (
    MAX_GPUS,
    MAX_NUM_BLOCKS_UPPER_BOUND,
    Signal,
    _multi_gpu_barrier,
    is_p2p_enabled,
)

# On AMD Systems, the loads from GLOBAL addressspace gives an improvement
# to the performance.
comptime _target_address_space = AddressSpace.GLOBAL if is_amd_gpu() else AddressSpace.GENERIC

comptime elementwise_epilogue_type = fn[
    dtype: DType, width: Int, *, alignment: Int
](Coord, SIMD[dtype, size=width]) capturing -> None


@always_inline
fn _load_reduce[
    dtype: DType,
    //,
    ngpus: Int,
    simd_width: Int,
    alignment: Int,
    accum_type: DType,
    *,
    use_multimem: Bool = False,
](
    elem_idx: Int,
    ptrs: InlineArray[
        UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
        1 if use_multimem else ngpus,
    ],
) -> SIMD[dtype, simd_width]:
    comptime if use_multimem:
        # Multimem mode: use optimized reduction
        return multimem_ld_reduce[
            dtype,
            simd_width=simd_width,
            reduction = ReduceOp.ADD,
            scope = Scope.GPU,
            consistency = Consistency.RELAXED,
            accum_type=accum_type,
        ]((ptrs[0] + elem_idx).address_space_cast[AddressSpace.GLOBAL]())
    else:
        # Regular mode: manual accumulation
        # Initialize with first load to avoid extra zero-add operation
        var accum = (
            ptrs[0]
            .address_space_cast[_target_address_space]()
            .load[width=simd_width, alignment=alignment, invariant=True](
                elem_idx
            )
            .cast[accum_type]()
        )

        comptime for gpu_idx in range(1, ngpus):
            accum += (
                ptrs[gpu_idx]
                .address_space_cast[_target_address_space]()
                .load[width=simd_width, alignment=alignment, invariant=True](
                    elem_idx
                )
                .cast[accum_type]()
            )

        return accum.cast[dtype]()


struct ReduceScatterConfig[
    dtype: DType,
    ngpus: Int,
    simd_width: Int = simd_width_of[dtype, target = get_gpu_target()](),
    alignment: Int = align_of[SIMD[dtype, simd_width]](),
    accum_type: DType = get_accum_type[dtype](),
](TrivialRegisterPassable):
    """Configuration for axis-aware reduce-scatter partitioning.

    Divides `axis_size` units evenly across GPUs. Lower ranks get one extra
    unit when there's a remainder. The 1D case is a special case where
    `axis_size = num_elements // simd_width` and `unit_numel = simd_width`.
    """

    var stride: Int
    var axis_part: Int
    var axis_remainder: Int
    var unit_numel: Int

    @always_inline
    fn __init__(
        out self,
        axis_size: Int,
        unit_numel: Int,
        threads_per_gpu: Int,
    ):
        """General constructor for axis-aware partitioning.

        Args:
            axis_size: Number of units along the scatter axis.
            unit_numel: Number of elements per unit.
            threads_per_gpu: Total threads per GPU.
        """
        comptime assert Self.ngpus > 1, "ngpus must be greater than 1"
        self.stride = threads_per_gpu * Self.simd_width
        self.axis_part = axis_size // Self.ngpus
        self.axis_remainder = axis_size % Self.ngpus
        self.unit_numel = unit_numel

    @always_inline
    fn __init__(
        out self,
        num_elements: Int,
        threads_per_gpu: Int,
    ):
        """1D convenience constructor. Partitions by SIMD vectors."""
        comptime assert Self.ngpus > 1, "ngpus must be greater than 1"
        self.stride = threads_per_gpu * Self.simd_width
        var num_simd_vectors = num_elements // Self.simd_width
        self.axis_part = num_simd_vectors // Self.ngpus
        self.axis_remainder = num_simd_vectors % Self.ngpus
        self.unit_numel = Self.simd_width

    @always_inline
    fn rank_unit_start(self, rank: Int) -> Int:
        """Start unit index along scatter axis for this rank."""
        return rank * self.axis_part + min(rank, self.axis_remainder)

    @always_inline
    fn rank_units(self, rank: Int) -> Int:
        """Number of units for this rank."""
        return self.axis_part + Int(rank < self.axis_remainder)

    @always_inline
    fn rank_num_elements(self, rank: Int) -> Int:
        """Total elements for this rank."""
        return self.rank_units(rank) * self.unit_numel

    @always_inline
    fn rank_start(self, rank: Int) -> Int:
        """Flat element start offset for this rank."""
        return self.rank_unit_start(rank) * self.unit_numel

    @always_inline
    fn rank_end(self, rank: Int) -> Int:
        """Flat element end offset for this rank."""
        return self.rank_start(rank + 1)

    @always_inline
    fn rank_part(self, rank: Int) -> Int:
        """Number of elements for this rank (alias for rank_num_elements)."""
        return self.rank_num_elements(rank)

    @always_inline
    fn thr_local_start(self, thread_idx: UInt) -> Int:
        return Int(thread_idx) * Self.simd_width


@always_inline
fn _reduce_scatter_flat_impl[
    dtype: DType,
    simd_width: Int,
    alignment: Int,
    accum_type: DType,
    //,
    ngpus: Int,
    *,
    output_lambda: elementwise_epilogue_type,
    use_multimem: Bool = False,
](
    src_ptrs: InlineArray[
        UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
        1 if use_multimem else ngpus,
    ],
    out_buf: TileTensor[mut=True, dtype, ...],
    my_rank: Int,
    config: ReduceScatterConfig[
        dtype, ngpus, simd_width, alignment, accum_type
    ],
):
    """Flat pointer-based reduce-scatter implementation.

    Used by allreduce's 2-stage kernel and multimem paths.
    """
    for idx in range(
        config.rank_start(my_rank) + config.thr_local_start(global_idx.x),
        config.rank_end(my_rank),
        config.stride,
    ):
        # float32 accumulator for numerical stability.
        var reduced_result = _load_reduce[
            ngpus,
            simd_width = config.simd_width,
            alignment = config.alignment,
            accum_type = config.accum_type,
            use_multimem=use_multimem,
        ](idx, src_ptrs)

        # Apply epilogue and store result.
        output_lambda[width = config.simd_width, alignment = config.alignment](
            out_buf.layout.idx2crd(idx - config.rank_start(my_rank)),
            reduced_result,
        )


@always_inline
fn _reduce_scatter_impl[
    dtype: DType,
    num_buffers: Int,
    in_tile_layout: TensorLayout,
    //,
    *,
    output_lambda: elementwise_epilogue_type,
    simd_width: Int = simd_width_of[dtype, target = get_gpu_target()](),
    alignment: Int = align_of[SIMD[dtype, simd_width]](),
    accum_type: DType = get_accum_type[dtype](),
](
    in_tiles: InlineArray[
        TileTensor[dtype, in_tile_layout, ImmutAnyOrigin],
        num_buffers,
    ],
    out_buf: TileTensor[mut=True, dtype, ...],
    num_elements: Int,
    thread_stride: Int,
):
    """TileTensor-based reduce-scatter implementation.

    Iterates flat over sliced+reversed input tiles with coalesced access.
    For 1D inputs: degenerates to contiguous pointer loads (same as flat).
    For 2D axis-0: also contiguous (reversed row-major = flat).
    For 2D axis-1: coalesced within stride-1 dimension strips.
    """
    for c in range(
        Int(global_idx.x) * simd_width,
        num_elements,
        thread_stride,
    ):
        var accum = (
            in_tiles[0]
            .address_space_cast[_target_address_space]()
            .load[width=simd_width, alignment=alignment, invariant=True](
                Coord(Idx(c))
            )
            .cast[accum_type]()
        )

        comptime for i in range(1, num_buffers):
            accum += (
                in_tiles[i]
                .address_space_cast[_target_address_space]()
                .load[width=simd_width, alignment=alignment, invariant=True](
                    Coord(Idx(c))
                )
                .cast[accum_type]()
            )

        output_lambda[width=simd_width, alignment=alignment](
            out_buf.layout.idx2crd(c),
            accum.cast[dtype](),
        )


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(BLOCK_SIZE))
)
fn _reducescatter_kernel[
    dtype: DType,
    in_layout: TensorLayout,
    out_layout: TensorLayout,
    ngpus: Int,
    *,
    axis: Int = -1,
    BLOCK_SIZE: Int,
    output_lambda: elementwise_epilogue_type,
    pdl_level: PDLLevel = PDLLevel(),
    use_multimem: Bool = False,
](
    in_bufs: InlineArray[
        TileTensor[dtype, in_layout, ImmutAnyOrigin],
        1 if use_multimem else ngpus,
    ],
    out_buf: TileTensor[dtype, out_layout, MutAnyOrigin],
    rank_sigs: InlineArray[UnsafePointer[Signal, MutAnyOrigin], MAX_GPUS],
    axis_size: Int,
    unit_numel: Int,
    my_rank: Int,
):
    """Reduce-scatter kernel with axis-aware slicing.

    Each GPU slices its partition from all input buffers, reverses the layout
    for coalesced access, reduces, and writes to its output.
    When use_multimem is True, uses hardware-accelerated multimem reduction.
    """
    comptime simd_width = simd_width_of[dtype, target = get_gpu_target()]()
    var my_sig = rank_sigs[my_rank]
    var threads_per_gpu = Int(grid_dim.x) * BLOCK_SIZE

    var config = ReduceScatterConfig[dtype, ngpus](
        axis_size, unit_numel, threads_per_gpu
    )

    comptime if pdl_level == PDLLevel.OVERLAP_AT_BEGINNING:
        launch_dependent_grids()

    comptime if pdl_level > PDLLevel.OFF:
        wait_on_dependent_grids()

    _multi_gpu_barrier[ngpus, is_start=True](rank_sigs, my_sig, my_rank)

    comptime if use_multimem:
        # Multimem: single pointer, hardware-accelerated reduction.
        var ptrs = InlineArray[UnsafePointer[Scalar[dtype], ImmutAnyOrigin], 1](
            uninitialized=True
        )
        ptrs[0] = in_bufs[0].ptr
        _reduce_scatter_flat_impl[
            ngpus, output_lambda=output_lambda, use_multimem=True
        ](ptrs, out_buf, my_rank, config)
    else:
        # Round-robin access pattern to balance NVLink traffic across GPUs.
        var reordered = InlineArray[
            TileTensor[dtype, in_layout, ImmutAnyOrigin], ngpus
        ](uninitialized=True)

        comptime for i in range(ngpus):
            reordered[i] = in_bufs[(my_rank + i) % ngpus]

        var u_start = config.rank_unit_start(my_rank)
        var n_units = config.rank_units(my_rank)
        var n_elements = config.rank_num_elements(my_rank)

        comptime if axis == -1:
            # Flat: construct sliced 1D tiles from input TileTensors (any rank).
            comptime FlatLayout = type_of(row_major(Idx(n_elements)))
            comptime FlatTile = TileTensor[dtype, FlatLayout, ImmutAnyOrigin]
            var flat_tiles = InlineArray[FlatTile, ngpus](uninitialized=True)
            var elem_start = u_start * config.unit_numel

            comptime for i in range(ngpus):
                flat_tiles[i] = FlatTile(
                    reordered[i].ptr + elem_start,
                    row_major(Idx(n_elements)),
                )

            _reduce_scatter_impl[output_lambda=output_lambda](
                flat_tiles, out_buf, n_elements, config.stride
            )
        else:
            # 2D axis-aware: slice + reverse for coalesced access.
            comptime InputTile = TileTensor[dtype, in_layout, ImmutAnyOrigin]
            comptime DynShapeTypes = _CoordToDynamic[
                InputTile.linear_idx_type, *in_layout._shape_types
            ]
            comptime RevLayout = Layout[
                Variadic.reverse[*DynShapeTypes],
                Variadic.reverse[*in_layout._stride_types],
            ]
            comptime SlicedRevTile = TileTensor[
                dtype, RevLayout, ImmutAnyOrigin
            ]
            var sliced_tiles = InlineArray[SlicedRevTile, ngpus](
                uninitialized=True
            )

            comptime if axis == 0:
                # Scatter along rows.
                var dim_1 = Int(reordered[0].dim[1]())
                comptime for i in range(ngpus):
                    var sliced = reordered[i].slice(
                        (u_start, u_start + n_units),
                        (0, dim_1),
                    )
                    sliced_tiles[i] = SlicedRevTile(
                        sliced.ptr, sliced.layout.reverse()
                    )
            else:
                # axis == 1: scatter along columns.
                var dim_0 = Int(reordered[0].dim[0]())
                var col_start = u_start * simd_width
                var col_end = col_start + n_units * simd_width
                comptime for i in range(ngpus):
                    var sliced = reordered[i].slice(
                        (0, dim_0),
                        (col_start, col_end),
                    )
                    sliced_tiles[i] = SlicedRevTile(
                        sliced.ptr, sliced.layout.reverse()
                    )

            _reduce_scatter_impl[output_lambda=output_lambda](
                sliced_tiles, out_buf, n_elements, config.stride
            )

    _multi_gpu_barrier[ngpus, is_start=False](rank_sigs, my_sig, my_rank)


@always_inline
fn _reducescatter_p2p[
    dtype: DType,
    ngpus: Int,
    in_layout: TensorLayout,
    in_origin: Origin,
    *,
    axis: Int = -1,
    output_lambda: elementwise_epilogue_type,
    pdl_level: PDLLevel = PDLLevel(),
    use_multimem: Bool = False,
](
    list_of_in_bufs: InlineArray[
        TileTensor[dtype, in_layout, in_origin],
        1 if use_multimem else ngpus,
    ],
    output_buffer: TileTensor[mut=True, dtype, ...],
    rank_sigs: InlineArray[UnsafePointer[Signal, MutAnyOrigin], MAX_GPUS],
    max_num_blocks: Int,
    ctx: DeviceContext,
    axis_size: Int,
    unit_numel: Int,
) raises:
    """Performs reducescatter using peer-to-peer access for a single GPU.

    Parameters:
        dtype: Data dtype of tensor elements.
        ngpus: Number of GPUs participating.
        in_layout: Layout of the input TileTensors.
        in_origin: Origin of the input TileTensors.
        axis: Scatter axis (-1 for flat 1D, 0 or 1 for 2D).
        output_lambda: Elementwise epilogue function to apply to reduced values.
        pdl_level: Control PDL behavior for the kernel.
        use_multimem: Whether multimem optimization is enabled.

    Args:
        list_of_in_bufs: Input buffers from all GPUs (peer access required).
        output_buffer: Output buffer for this GPU's partition of reduced data.
        rank_sigs: Signal pointers for synchronization.
        max_num_blocks: Maximum number of thread blocks to launch.
        ctx: Device context for THIS GPU.
        axis_size: Number of units along the scatter axis.
        unit_numel: Number of elements per unit.
    """
    comptime simd_width = simd_width_of[dtype, target = get_gpu_target()]()
    comptime BLOCK_SIZE = 256
    comptime num_buffers = 1 if use_multimem else ngpus

    # Grid size based on max per-GPU elements (rank 0 has most).
    var config_for_grid = ReduceScatterConfig[dtype, ngpus](
        axis_size, unit_numel, 0
    )
    var max_rank_elements = config_for_grid.rank_num_elements(0)

    # We guard against max_rank_elements % simd_width != 0 in reducescatter
    var grid_size = min(
        max_num_blocks,
        ceildiv(max_rank_elements // simd_width, BLOCK_SIZE),
    )

    # Erase origin to ImmutAnyOrigin for the kernel.
    # TODO(KERN-2526): is this necessary?
    comptime KernelInputType = TileTensor[dtype, in_layout, ImmutAnyOrigin]
    var kernel_in_bufs = InlineArray[KernelInputType, num_buffers](
        uninitialized=True
    )
    comptime for i in range(num_buffers):
        kernel_in_bufs[i] = KernelInputType(
            list_of_in_bufs[i].ptr, list_of_in_bufs[i].layout
        )

    comptime kernel = _reducescatter_kernel[
        dtype,
        in_layout,
        output_buffer.LayoutType,
        ngpus,
        axis=axis,
        BLOCK_SIZE=BLOCK_SIZE,
        output_lambda=output_lambda,
        pdl_level=pdl_level,
        use_multimem=use_multimem,
    ]

    # Launch the kernel
    ctx.enqueue_function[kernel, kernel](
        kernel_in_bufs,
        output_buffer,
        rank_sigs,
        axis_size,
        unit_numel,
        Int(ctx.id()),
        grid_dim=grid_size,
        block_dim=BLOCK_SIZE,
    )


@parameter
fn reducescatter[
    dtype: DType,
    ngpus: Int,
    in_layout: TensorLayout,
    in_origin: Origin,
    output_lambda: Optional[elementwise_epilogue_type] = None,
    pdl_level: PDLLevel = PDLLevel(),
    *,
    axis: Int = -1,
    use_multimem: Bool = False,
](
    input_buffers: InlineArray[
        TileTensor[dtype, in_layout, in_origin],
        1 if use_multimem else ngpus,
    ],
    output_buffer: TileTensor[mut=True, dtype, ...],
    rank_sigs: InlineArray[UnsafePointer[Signal, MutAnyOrigin], MAX_GPUS],
    ctx: DeviceContext,
    _max_num_blocks: Optional[Int] = None,
) raises:
    """Per-device reducescatter operation with axis-aware scatter.

    Performs a reduce-scatter across multiple GPUs: each GPU reduces its assigned
    partition from all input buffers and writes the result to its output buffer.

    Parameters:
        dtype: Data dtype of tensor elements.
        ngpus: Number of GPUs participating.
        in_layout: Layout of the input TileTensors.
        in_origin: Origin of the input TileTensors.
        output_lambda: Optional elementwise epilogue function. If not provided,
            reduced values are stored directly to output_buffer.
        pdl_level: Control PDL behavior for the kernel.
        axis: Scatter axis. -1 for flat 1D partitioning (default).
            0 to scatter along rows, 1 to scatter along columns.
            Requires 2D row-major inputs when axis >= 0.
        use_multimem: If True, use hardware-accelerated multimem reduction.
            Only valid with axis=-1 (flat partitioning).

    Args:
        input_buffers: Input TileTensors from all GPUs (peer access required).
            When use_multimem is True, a single multimem-mapped TileTensor.
        output_buffer: Output TileTensor for THIS GPU's partition of reduced data.
        rank_sigs: Signal pointers for synchronization between GPUs.
        ctx: Device context for THIS GPU.
        _max_num_blocks: Optional maximum number of thread blocks to launch.
            If not specified, uses MAX_NUM_BLOCKS_UPPER_BOUND.

    Raises:
        Error: If P2P access is not available between GPUs.
        Error: If input buffer size is not a multiple of SIMD width.
    """
    comptime assert ngpus >= 2, "reducescatter requires at least 2 GPUs"
    comptime simd_width = simd_width_of[dtype, target = get_gpu_target()]()

    # Validate axis and rank combination.
    # TODO(KERN-2526): generalize to higher dims & multimem support
    comptime if axis >= 0:
        comptime assert in_layout.rank == 2, "axis >= 0 requires 2D input"
        comptime assert axis == 0 or axis == 1, "axis must be -1, 0, or 1"
    comptime if use_multimem:
        comptime assert (
            axis == -1
        ), "use_multimem only supported with axis=-1 (flat)"

    # Return early if the input buffer is empty
    var num_elements = input_buffers[0].numel()
    if num_elements == 0:
        return

    if not is_p2p_enabled():
        raise Error("Reducescatter currently requires P2P access between GPUs")

    # Compute axis_size and unit_numel based on axis.
    var axis_size: Int
    var unit_numel: Int
    comptime if axis == -1:
        # 1D: partition by SIMD vectors
        if num_elements % simd_width != 0:
            raise Error(
                "non SIMD-width multiple number of elements unsupported by"
                " reducescatter"
            )
        axis_size = num_elements // simd_width
        unit_numel = simd_width
    elif axis == 0:
        # 2D axis-0: partition rows, unit = one row
        var dim_0 = Int(input_buffers[0].layout.shape[0]().value())
        var dim_1 = Int(input_buffers[0].layout.shape[1]().value())
        if dim_1 % simd_width != 0:
            raise Error(
                "inner dimension (axis 1) must be a multiple of SIMD width"
                " for axis-0 reduce-scatter"
            )
        axis_size = dim_0
        unit_numel = dim_1
    else:
        # axis == 1: partition column groups, unit = simd_width columns
        var dim_0 = Int(input_buffers[0].layout.shape[0]().value())
        var dim_1 = Int(input_buffers[0].layout.shape[1]().value())
        if dim_1 % simd_width != 0:
            raise Error(
                "scatter dimension (axis 1) must be a multiple of SIMD width"
                " for axis-1 reduce-scatter"
            )
        axis_size = dim_1 // simd_width
        unit_numel = dim_0 * simd_width

    # Validate output buffer shape for this rank's partition.
    var my_rank = Int(ctx.id())
    var config_check = ReduceScatterConfig[dtype, ngpus](
        axis_size, unit_numel, 0
    )
    var expected_numel = config_check.rank_num_elements(my_rank)
    comptime if axis == -1:
        if output_buffer.numel() != expected_numel:
            raise Error(
                "output buffer has "
                + String(output_buffer.numel())
                + " elements, expected "
                + String(expected_numel)
            )
    else:
        comptime assert (
            output_buffer.rank == 2
        ), "axis >= 0 requires 2D output buffer"
        var n_units = config_check.rank_units(my_rank)
        var expected_rows = n_units if axis == 0 else Int(
            input_buffers[0].layout.shape[0]().value()
        )
        var expected_cols = (
            Int(input_buffers[0].layout.shape[1]().value()) if axis
            == 0 else n_units * simd_width
        )
        var out_rows = Int(output_buffer.dim[0]())
        var out_cols = Int(output_buffer.dim[1]())
        if out_rows != expected_rows or out_cols != expected_cols:
            raise Error(
                "output buffer shape ("
                + String(out_rows)
                + ", "
                + String(out_cols)
                + "), expected ("
                + String(expected_rows)
                + ", "
                + String(expected_cols)
                + ")"
            )

    var max_num_blocks = (
        _max_num_blocks.value() if _max_num_blocks else MAX_NUM_BLOCKS_UPPER_BOUND
    )

    # Default epilogue: store directly to output buffer
    @always_inline
    @parameter
    @__copy_capture(output_buffer)
    fn default_output_lambda[
        _dtype: DType,
        _width: Int,
        *,
        _alignment: Int,
    ](coords: Coord, val: SIMD[_dtype, _width]) -> None where (
        coords.flat_rank == output_buffer.flat_rank
    ):
        output_buffer.store[width=_width, alignment=_alignment](
            coords, val.cast[dtype]()
        )

    comptime actual_output_lambda = default_output_lambda if not output_lambda else output_lambda.value()

    # Launch the reduce-scatter kernel via P2P
    _reducescatter_p2p[
        dtype,
        ngpus,
        axis=axis,
        output_lambda=actual_output_lambda,
        pdl_level=pdl_level,
        use_multimem=use_multimem,
    ](
        input_buffers,
        output_buffer,
        rank_sigs,
        max_num_blocks,
        ctx,
        axis_size,
        unit_numel,
    )
