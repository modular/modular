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
"""Multi-GPU allreduce implementation for efficient tensor reduction across GPUs.

This module provides an optimized implementation of allreduce operations across multiple GPUs,
supporting both peer-to-peer (P2P) and non-P2P communication patterns. The implementation
automatically selects between two approaches based on hardware capabilities:

1. P2P-based implementation (when P2P access is available):
   - Uses direct GPU-to-GPU memory access for better performance
   - Implements both single-stage and two-stage algorithms:
     - Single-stage for latency-bound transfers (small tensors)
     - Two-stage (reduce-scatter + all-gather) for bandwidth-bound transfers (large tensors)
   - Optimized for NVLink bandwidth utilization
   - Uses vectorized memory access and higher precision accumulation

2. Non-P2P fallback implementation:
   - Copies data through host memory when direct GPU access isn't possible
   - Simple but functional approach for systems without P2P support

The implementation is tuned for common GPU architectures (A100, H100) and includes
parameters that can be adjusted for different hardware configurations.

## Per-Device Architecture

The allreduce operation follows a per-device execution model:

1. **Single-Device Instances**: Each GPU runs its own instance of the allreduce
   operation.

2. **Parallel Execution**: The Python/Graph API layer is responsible for:
   - Creating one allreduce op instance per participating GPU.
   - Ensuring all instances execute in parallel.
   - Ensuring correctness by staging mo.fence.

3. **Device Affinity**: Each allreduce instance:
   - Executes on its assigned GPU (specified via device context).
   - Reads from all GPUs' input buffers (requires P2P access).
   - Writes only to its own output buffer.
   - Uses the same synchronization signals as other instances.

4. **Requirements**:
   - Peer-to-peer access must be enabled between all participating GPUs.
   - All instances must launch before any can complete (for synchronization).
   - The device context determines which GPU executes each instance.

Limitations:
- Maximum of 8 GPUs supported.
- Multimem mode still requires the element count to be a multiple of SIMD width.
- All input/output buffers must have identical shapes.

Non-multimem 1-stage P2P and naive epilogue accept arbitrary ``N``: when
``N`` is a multiple of device SIMD width, the 1-stage kernel uses the same
vectorized ``_load_reduce`` grid loop as before; otherwise it runs that loop on
the SIMD-aligned prefix and finishes the last ``< simd_width`` elements with a
grid-strided scalar reduce-store. The naive epilogue kernel uses the same
SIMD-prefix + scalar-tail pattern for ``accum → out``.

## Visual Overview

1) 1-Stage P2P (latency-bound)

   Each GPU r reads its portion from every peer buffer directly (via P2P),
   accumulates, then writes to its result using the epilogue:

       GPU r (result_r)
       src_tensors[0] ─┐
       src_tensors[1] ─┼──► Σ (high-precision accum) ──► output_lambda ──► result_r
       ...         ─┘

   Notes:
   - Non-multimem: SIMD-vector ``_load_reduce`` on the aligned prefix; optional
     scalar tail when ``N`` is not a multiple of SIMD width. Multimem: unchanged
     full-vector loads (``N`` must be SIMD-aligned).
   - Good for small/latency-bound tensors.

2) 2-Stage P2P (bandwidth-bound)

   Stage 1 (reduce-scatter): Each GPU r reduces its assigned partition and writes
   into its own signal payload (the bytes after the Signal header).

       src_tensors[*]  ──►  reduce(partition r)  ──►  rank_sigs[r].payload  (per-GPU)

   Stage 2 (all-gather): Each GPU r gathers all partitions from peers' payloads
   and writes them to its result using the epilogue.

       [payload_0], [payload_1], ..., [payload_{ngpus-1}]  ──►  result_r (via output_lambda)

For the naive allreduce (no P2P) per-device flow and staging details, see the
`_allreduce_naive_single` docstring in this file.
"""

from std.collections import InlineArray
from std.math import ceildiv
from std.sys import align_of, simd_width_of, size_of

from layout import Coord, Idx, TileTensor, row_major
from layout.tile_layout import TensorLayout
from std.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    block_dim,
    global_idx,
    grid_dim,
)
from std.gpu.primitives.grid_controls import (
    PDL,
    PDLLevel,
    pdl_launch_attributes,
)
from std.gpu.host import DeviceBuffer, DeviceContext, get_gpu_target

from std.utils import StaticTuple
from std.utils.numerics import get_accum_type

from std.collections.optional import Optional

from .reducescatter import (
    ReduceScatterConfig,
    _reduce_scatter_impl,
    _load_reduce,
    _target_address_space,
)
from .sync import (
    MAX_GPUS,
    MAX_NUM_BLOCKS_UPPER_BOUND,
    Signal,
    _multi_gpu_barrier,
    circular_add,
    is_p2p_enabled,
)
from .device_query import dispatch_max_num_blocks, CommTuningConfig
from internal_utils import Table

comptime elementwise_epilogue_type = def[
    dtype: DType, width: SIMDSize, *, alignment: Int
](Coord, SIMD[dtype, size=width]) capturing -> None

# Tuning table to get num_blocks for allreduce.
# Arch-specific defaults use ngpus=-1, num_bytes=-1 with the arch's sm_version.
# The global default (sm_version="default") is the ultimate fallback for
# unknown architectures -- dispatch_max_num_blocks prefers arch-specific
# defaults when available.
comptime allreduce_tuning_table = Table(
    [
        # default for sm90 (encoded with ngpus=-1, num_bytes=-1)
        CommTuningConfig(
            ngpus=-1, num_bytes=-1, sm_version="sm_90a", num_blocks=216
        ),
        CommTuningConfig(
            ngpus=4, num_bytes=(1 << 27), sm_version="sm_90a", num_blocks=232
        ),
        # default for sm100 (encoded with ngpus=-1, num_bytes=-1)
        CommTuningConfig(
            ngpus=-1, num_bytes=-1, sm_version="sm_100a", num_blocks=512
        ),
        # Tuning results for sm100 (2xB200, 4xB200)
        CommTuningConfig(
            ngpus=2, num_bytes=(1 << 23), sm_version="sm_100a", num_blocks=512
        ),
        CommTuningConfig(
            ngpus=2, num_bytes=(1 << 24), sm_version="sm_100a", num_blocks=512
        ),
        CommTuningConfig(
            ngpus=2, num_bytes=(1 << 25), sm_version="sm_100a", num_blocks=512
        ),
        CommTuningConfig(
            ngpus=2, num_bytes=(1 << 26), sm_version="sm_100a", num_blocks=512
        ),
        CommTuningConfig(
            ngpus=2, num_bytes=(1 << 27), sm_version="sm_100a", num_blocks=512
        ),
        CommTuningConfig(
            ngpus=4, num_bytes=(1 << 23), sm_version="sm_100a", num_blocks=512
        ),
        CommTuningConfig(
            ngpus=4, num_bytes=(1 << 24), sm_version="sm_100a", num_blocks=512
        ),
        CommTuningConfig(
            ngpus=4, num_bytes=(1 << 25), sm_version="sm_100a", num_blocks=512
        ),
        CommTuningConfig(
            ngpus=4, num_bytes=(1 << 26), sm_version="sm_100a", num_blocks=512
        ),
        CommTuningConfig(
            ngpus=4, num_bytes=(1 << 27), sm_version="sm_100a", num_blocks=512
        ),
        # default for sm103 (B300, encoded with ngpus=-1, num_bytes=-1)
        CommTuningConfig(
            ngpus=-1, num_bytes=-1, sm_version="sm_103a", num_blocks=512
        ),
        # default for CDNA3 (MI300X, encoded with ngpus=-1, num_bytes=-1)
        CommTuningConfig(
            ngpus=-1, num_bytes=-1, sm_version="CDNA3", num_blocks=32
        ),
        # default for CDNA4 (MI355X, encoded with ngpus=-1, num_bytes=-1)
        CommTuningConfig(
            ngpus=-1, num_bytes=-1, sm_version="CDNA4", num_blocks=64
        ),
        CommTuningConfig(
            ngpus=8, num_bytes=(1 << 20), sm_version="CDNA4", num_blocks=64
        ),
        CommTuningConfig(
            ngpus=8, num_bytes=(1 << 31), sm_version="CDNA4", num_blocks=44
        ),
        # global default for unknown architectures
        CommTuningConfig(
            ngpus=-1, num_bytes=-1, sm_version="default", num_blocks=512
        ),
    ],
    "allreduce_table",
)


@__name(t"naive_reduce_{dtype}", mangle=True)
def _naive_reduce_kernel[
    dtype: DType
](
    dst_buf: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    src_buf: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    num_elements: Int,
):
    """
    A simple reduction kernel that adds source buffer values to destination buffer.

    Parameters:
        dtype: DType - The data type of the values being reduced.

    Args:
        dst_buf: Destination buffer to accumulate results.
        src_buf: Source buffer containing values to add.
        num_elements: Number of elements to process.

    Each thread handles multiple elements with striding for coalesced memory access.
    """
    var tid = global_idx.x
    var stride = grid_dim.x * block_dim.x

    # Each thread handles multiple elements with striding
    for i in range(tid, num_elements, stride):
        dst_buf[i] += src_buf[i]


@__name(t"naive_reduce_with_lambda_{dtype}", mangle=True)
def _naive_reduce_kernel_with_lambda[
    dtype: DType,
    out_layout: TensorLayout,
    *,
    output_lambda: elementwise_epilogue_type,
](
    dst_buf: TileTensor[dtype, out_layout, MutAnyOrigin],
    src_buf: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    num_elements: Int,
):
    """Apply ``output_lambda`` from ``src_buf`` into ``dst_buf`` (naive epilogue).

    Uses device SIMD width loads on the aligned prefix (same pattern as the
    pre-ragged vector epilogue), then grid-strided scalar loads for any tail when
    ``num_elements`` is not a multiple of SIMD width.
    """
    comptime simd_width = simd_width_of[dtype, target=get_gpu_target()]()
    comptime simd_align = align_of[SIMD[dtype, simd_width]]()
    comptime scalar_align = align_of[SIMD[dtype, 1]]()
    var global_tid = global_idx.x
    var total_threads = grid_dim.x * Int(block_dim.x)
    var num_simd_vectors = num_elements // simd_width
    var simd_prefix_elems = num_simd_vectors * simd_width

    if num_simd_vectors > 0:
        for idx in range(global_tid, num_simd_vectors, total_threads):
            var elem_idx = idx * simd_width
            output_lambda[width=simd_width, alignment=simd_align](
                dst_buf.layout.idx2crd(elem_idx),
                src_buf.load[width=simd_width, alignment=simd_align](elem_idx),
            )

    if simd_prefix_elems < num_elements:
        for elem_idx in range(
            simd_prefix_elems + global_tid, num_elements, total_threads
        ):
            output_lambda[width=1, alignment=scalar_align](
                dst_buf.layout.idx2crd(elem_idx),
                src_buf.load[width=1, alignment=scalar_align](elem_idx),
            )


@always_inline
def _allreduce_naive_single[
    dtype: DType,
    ngpus: Int,
    in_layout: TensorLayout,
    in_origin: Origin,
    out_layout: TensorLayout,
    output_lambda: elementwise_epilogue_type,
    num_tensors: Int = ngpus,
](
    list_of_in_tensors: InlineArray[
        TileTensor[dtype, in_layout, in_origin], num_tensors
    ],
    out_tensor: TileTensor[mut=True, dtype, out_layout, ...],
    max_num_blocks: Int,
    ctx: DeviceContext,
) raises:
    """Naive per-device allreduce using a local temporary staging buffer.

    Overview
    - One op instance runs per GPU ("device r").
    - Each instance builds its local result by summing all inputs into a local
      accumulation buffer, then writes to its own output.
    - To stage remote inputs for accumulation (no P2P), it allocates a temporary
      buffer on the current device.

    Memory layout per device (r):

        tmp_r  (device-local buffer, length = N elements)

    Parameters:
        dtype: The data type of tensor elements.
        ngpus: Number of GPUs participating in allreduce.
        in_layout: Layout of the input TileTensors.
        in_origin: Origin of the input TileTensors.
        out_layout: Layout of the output TileTensor.
        output_lambda: An elementwise output lambda function.
        num_tensors: Number of buffers to process (defaults to ngpus).

    Per-device flow (device r):

        in_r  ───────►  accumulate into A_r
        for each i != r:
          in_i  ──copy──►  S_r  ──accumulate──►  A_r
        A_r  ──output_lambda──► out_r

    ASCII for a 3-GPU example (naive path, no P2P):

        GPU0:  in0  →  A0 += in0
               in1  →  tmp0 → A0 += tmp0
               in2  →  tmp0 → A0 += tmp0
               A0   →  out0 (via output_lambda)

        GPU1:  in1  →  A1 += in1
               in0  →  tmp1 → A1 += tmp1
               in2  →  tmp1 → A1 += tmp1
               A1   →  out1 (via output_lambda)

        GPU2:  in2  →  A2 += in2
               in0  →  tmp2 → A2 += tmp2
               in1  →  tmp2 → A2 += tmp2
               A2   →  out2 (via output_lambda)

    Requirements
    - Inputs across GPUs must be identical shape and dtype.
    - Each op instance only writes to its own temporary buffer and its own
      output buffer (`out_r`).
    """
    comptime BLOCK_SIZE = 256
    var num_elements = list_of_in_tensors[0].num_elements()

    # Wrap ALL input buffers as DeviceBuffer with their respective device contexts.
    # rebind to MutAnyOrigin is safe: DeviceBuffer only reads via DMA copy.
    var dev_inputs = List[DeviceBuffer[dtype]](capacity=ngpus)
    for i in range(ngpus):
        var rctx = DeviceContext(device_id=i)
        dev_inputs.append(
            DeviceBuffer[dtype](
                rctx,
                rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](
                    list_of_in_tensors[i].ptr
                ),
                num_elements,
                owning=False,
            )
        )

    # Accumulation buffer on this device.
    var accum = ctx.enqueue_create_buffer[dtype](num_elements)
    ctx.enqueue_memset(accum, 0)

    # Resolve this device's rank and allocate a temp staging buffer.
    var my_rank: Int = Int(ctx.id())
    var scratch = ctx.enqueue_create_buffer[dtype](num_elements)

    # Grid configuration for naive kernels.
    var grid_size = min(max_num_blocks, ceildiv(num_elements, BLOCK_SIZE))
    comptime simd_width_epi = simd_width_of[dtype, target=get_gpu_target()]()
    var num_simd_vecs_epi = num_elements // simd_width_epi
    var tail_elems_epi = num_elements - num_simd_vecs_epi * simd_width_epi
    var grid_simd_epi = ceildiv(num_simd_vecs_epi, BLOCK_SIZE)
    var grid_tail_epi = ceildiv(tail_elems_epi, BLOCK_SIZE)
    var grid_epilogue = min(
        max_num_blocks,
        max(max(grid_simd_epi, grid_tail_epi), 1),
    )

    # Reduce local buffer first.
    ctx.enqueue_function[
        _naive_reduce_kernel[dtype], _naive_reduce_kernel[dtype]
    ](
        accum,
        dev_inputs[my_rank],
        num_elements,
        grid_dim=grid_size,
        block_dim=BLOCK_SIZE,
    )

    # Reduce contributions from peers via scratch.
    for i in range(ngpus):
        if i == my_rank:
            continue

        # Copy remote input into device-local scratch, then accumulate.
        ctx.enqueue_copy(scratch, dev_inputs[i])
        ctx.enqueue_function[
            _naive_reduce_kernel[dtype], _naive_reduce_kernel[dtype]
        ](
            accum,
            scratch,
            num_elements,
            grid_dim=grid_size,
            block_dim=BLOCK_SIZE,
        )

    # Apply elementwise epilogue to write into the output buffer.
    comptime naive_reduce_with_lambda_kernel = _naive_reduce_kernel_with_lambda[
        dtype,
        out_layout,
        output_lambda=output_lambda,
    ]
    ctx.enqueue_function[
        naive_reduce_with_lambda_kernel, naive_reduce_with_lambda_kernel
    ](
        rebind[TileTensor[dtype, out_layout, MutAnyOrigin]](out_tensor),
        accum,
        num_elements,
        grid_dim=grid_epilogue,
        block_dim=BLOCK_SIZE,
    )


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(BLOCK_SIZE))
)
@__name(t"allreduce_2stage_{dtype}_{use_multimem}", mangle=True)
def _allreduce_2stage_kernel[
    dtype: DType,
    ngpus: Int,
    in_layout: TensorLayout,
    out_layout: TensorLayout,
    *,
    BLOCK_SIZE: Int,
    output_lambda: elementwise_epilogue_type,
    use_multimem: Bool = False,
](
    result: TileTensor[dtype, out_layout, MutAnyOrigin],
    src_tensors: InlineArray[
        TileTensor[dtype, in_layout, ImmutAnyOrigin],
        1 if use_multimem else ngpus,
    ],
    rank_sigs: InlineArray[UnsafePointer[Signal, MutAnyOrigin], MAX_GPUS],
    num_elements: Int,
    my_rank: Int,
):
    """2-stage allreduce algorithm for bandwidth-bound transfers.

    This kernel implements a reduce-scatter + all-gather algorithm that is
    bandwidth optimal.

    Parameters:
        dtype: Data dtype of tensor elements.
        ngpus: Number of GPUs participating.
        in_layout: Layout of the input TileTensors.
        out_layout: Layout of the output TileTensor.
        BLOCK_SIZE: Number of threads per block.
        output_lambda: An elementwise output lambda function.
        use_multimem: If True, use multi-memory space buffers for input.

    Args:
        result: Output buffer for reduced values.
        src_tensors: Input buffers from all GPUs.
        rank_sigs: Signal pointers for synchronization.
            IMPORTANT: the Signal pointers have trailing buffers for
            communication, which must be at least `ngpus * size_of(payload)`.
            | -- size_of(Signal) -- | ------ a few MB ----- |
        num_elements: Number of elements to reduce.
        my_rank: Current GPU rank.
    """
    var my_sig = rank_sigs[my_rank]

    # --- Thread Indexing ---
    var global_tid = global_idx.x
    # Stride equals total threads in grid dimension for grid-strided loops.
    var stride = grid_dim.x * BLOCK_SIZE

    var rs_config = ReduceScatterConfig[dtype, ngpus](num_elements, stride)

    comptime num_tensors = 1 if use_multimem else ngpus

    with PDL():
        # --- Define tmp buffers by offsetting for Signal struct ---
        var tmps = InlineArray[
            UnsafePointer[Scalar[dtype], MutAnyOrigin], ngpus
        ](uninitialized=True)

        comptime for i in range(ngpus):
            # Round-robin access pattern to balance NVLink traffic across GPUs.
            var target = circular_add[ngpus](my_rank, i)
            # Skip Signal header.
            tmps[i] = (
                rank_sigs[target].address_space_cast[AddressSpace.GENERIC]() + 1
            ).bitcast[Scalar[dtype]]()

        # Current rank's output buffer.
        var tmp_out = tmps[0]

        # --- Stage 1: Reduce-Scatter Phase ---
        # Uses two-phase synchronization protocol with release-acquire semantics:
        # 1. Initial barrier establishes happens-before relationship.
        # 2. Memory fence ensures visibility of partial reductions.
        _multi_gpu_barrier[ngpus, is_start=True](rank_sigs, my_sig, my_rank)

        # TODO(KERN-2273): Remove this once temporary buffers removed
        # Output lambda for reduce-scatter: write to scratch buffer
        var tmp_buff = TileTensor[mut=True, dtype](
            tmp_out, row_major(Idx(rs_config.rank_part(my_rank)))
        )

        @always_inline
        @parameter
        @__copy_capture(tmp_buff)
        def rs_output_lambda[
            _dtype: DType,
            _width: SIMDSize,
            *,
            _alignment: Int,
        ](coords: Coord, val: SIMD[_dtype, _width]) -> None:
            tmp_buff.address_space_cast[_target_address_space]().store[
                width=_width, alignment=_alignment
            ](
                coords,
                val.cast[dtype](),
            )

        # Slice input tiles to this rank's partition for reduce-scatter.
        var elem_start = rs_config.rank_start(my_rank)
        var n_elements = rs_config.rank_num_elements(my_rank)
        comptime SlicedTile = TileTensor[dtype, SlicedLayout, ImmutAnyOrigin]
        comptime SlicedLayout = type_of(row_major(Idx(n_elements)))
        var sliced_tiles = InlineArray[SlicedTile, num_tensors](
            uninitialized=True
        )

        comptime for i in range(num_tensors):
            # Round-robin access pattern to balance NVLink traffic across GPUs.
            var target = 0 if num_tensors == 1 else circular_add[num_tensors](
                my_rank, i
            )
            sliced_tiles[i] = SlicedTile(
                src_tensors[target].ptr + elem_start, row_major(Idx(n_elements))
            )

        _reduce_scatter_impl[
            ngpus, output_lambda=rs_output_lambda, use_multimem=use_multimem
        ](sliced_tiles, tmp_buff, n_elements, rs_config.stride)

        # Second barrier with memory ordering guarantees.
        _multi_gpu_barrier[ngpus, is_start=False, need_fence=True](
            rank_sigs, my_sig, my_rank
        )

        # --- Stage 2: All-Gather Phase ---
        # Maintains thread index consistency to satisfy memory model:
        # The same tid guarantees visibility of prior writes.
        # So if thread `idx` computes the sum of `start + idx` in the first stage,
        # then thread `idx` also gathers `start + idx` from all ranks.
        comptime simd_width = rs_config.simd_width
        comptime alignment = rs_config.alignment

        # Ragged handling:
        # GPU-0 is guaranteed to have largest partition
        # GPU-ngpus-1 has smallest partition (only 1 simd vector smaller)

        # Main loop - only process unragged elements (no bounds check)
        for idx in range(
            rs_config.thr_local_start(global_tid),
            rs_config.rank_part(ngpus - 1),
            rs_config.stride,
        ):
            comptime for gpu_idx in range(ngpus):
                var peer_rank = circular_add[ngpus](my_rank, gpu_idx)

                var dst_idx = rs_config.rank_start(peer_rank) + idx
                output_lambda[width=simd_width, alignment=alignment](
                    result.layout.idx2crd(dst_idx),
                    tmps[gpu_idx]
                    .address_space_cast[_target_address_space]()
                    .load[width=simd_width, alignment=alignment](idx),
                )

        # Ragged tail - max 1 simd vector per gpu, spread work between threads
        if global_tid < ngpus:
            var peer_rank = circular_add[ngpus](my_rank, global_tid)
            if peer_rank < rs_config.axis_remainder:
                var idx = (
                    rs_config.rank_part(0) - simd_width
                )  # last ragged simd_vector
                var dst_idx = rs_config.rank_start(peer_rank) + idx
                output_lambda[width=simd_width, alignment=alignment](
                    result.layout.idx2crd(dst_idx),
                    tmps[global_tid]
                    .address_space_cast[_target_address_space]()
                    .load[width=simd_width, alignment=alignment](idx),
                )


@always_inline
def _allreduce_1stage_reduce_store_one[
    dtype: DType,
    in_layout: TensorLayout,
    out_layout: TensorLayout,
    num_tensors: Int,
    *,
    accum_type: DType,
    output_lambda: elementwise_epilogue_type,
](
    elem_idx: Int,
    ptrs: InlineArray[
        TileTensor[dtype, in_layout, ImmutAnyOrigin], num_tensors
    ],
    result: TileTensor[dtype, out_layout, MutAnyOrigin],
) -> None:
    """Load one element from every peer, reduce in ``accum_type``, epilogue store.
    """
    comptime scalar_align = align_of[SIMD[dtype, 1]]()
    var accum = (
        ptrs[0]
        .address_space_cast[_target_address_space]()
        .load[width=1, alignment=scalar_align, invariant=True](
            Coord(Idx(elem_idx))
        )
        .cast[accum_type]()
    )
    comptime for gpu_idx in range(1, num_tensors):
        accum += (
            ptrs[gpu_idx]
            .address_space_cast[_target_address_space]()
            .load[width=1, alignment=scalar_align, invariant=True](
                Coord(Idx(elem_idx))
            )
            .cast[accum_type]()
        )
    var reduced = accum.cast[dtype]()
    output_lambda[width=1, alignment=scalar_align](
        result.layout.idx2crd(elem_idx),
        reduced,
    )


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(BLOCK_SIZE))
)
@__name(t"allreduce_1stage_{dtype}_{use_multimem}", mangle=True)
def _allreduce_1stage_kernel[
    dtype: DType,
    ngpus: Int,
    in_layout: TensorLayout,
    out_layout: TensorLayout,
    *,
    BLOCK_SIZE: Int,
    output_lambda: elementwise_epilogue_type,
    use_multimem: Bool = False,
](
    result: TileTensor[dtype, out_layout, MutAnyOrigin],
    src_tensors: InlineArray[
        TileTensor[dtype, in_layout, ImmutAnyOrigin],
        1 if use_multimem else ngpus,
    ],
    rank_sigs: InlineArray[UnsafePointer[Signal, MutAnyOrigin], MAX_GPUS],
    num_elements: Int,
    my_rank: Int,
):
    """
    Kernel implementing allreduce using peer-to-peer access between GPUs.

    Parameters:
        dtype: Data dtype of tensor elements.
        ngpus: Number of GPUs participating.
        in_layout: Layout of the input TileTensors.
        out_layout: Layout of the output TileTensor.
        BLOCK_SIZE: Number of threads per block.
        output_lambda: An elementwise output lambda function.
        use_multimem: If True, use multi-memory space buffers for input.

    Args:
        result: Output tensor for reduced values
        src_tensors: Input tensors from all GPUs
        rank_sigs: Signal pointers for synchronization
        num_elements: Number of elements to reduce
        my_rank: Current GPU rank

    Uses P2P access to directly read from other GPU buffers and perform reduction.
    Synchronizes using _multi_gpu_barrier before and after reduction.

    **Non-multimem path:** grid-strided loop over full SIMD vectors using
    ``_load_reduce`` (same as historical 1-stage performance). If
    ``num_elements`` is not divisible by ``simd_width``, the aligned prefix is
    still processed as SIMD vectors; the remaining ``< simd_width`` scalars use
    ``_allreduce_1stage_reduce_store_one``.

    **Multimem path:** unchanged vectorized SIMD loads over full
    ``simd_width`` vectors (input size must remain SIMD-aligned).
    """
    comptime accum_type = get_accum_type[dtype]()
    comptime simd_width = simd_width_of[dtype, target=get_gpu_target()]()
    comptime alignment = align_of[SIMD[dtype, simd_width]]()

    var global_tid = global_idx.x
    var total_threads = grid_dim.x * BLOCK_SIZE
    var my_sig = rank_sigs[my_rank]

    # Route input pointers according to round-robin pattern.
    # For 8 GPUs: Rank 0 accesses 0→1→2→...→7, Rank 1 accesses 1→2→...→7→0, etc.
    comptime num_tensors = 1 if use_multimem else ngpus
    var ptrs = InlineArray[
        TileTensor[dtype, in_layout, ImmutAnyOrigin], num_tensors
    ](uninitialized=True)

    # It's safe to prefetch the input pointers
    comptime for i in range(num_tensors):
        var target = 0 if num_tensors == 1 else circular_add[num_tensors](
            my_rank, i
        )
        ptrs[i] = src_tensors[target]

    with PDL():
        _multi_gpu_barrier[ngpus, is_start=True](rank_sigs, my_sig, my_rank)

        comptime if use_multimem:
            var num_simd_chunks = num_elements // simd_width
            for idx in range(global_tid, num_simd_chunks, total_threads):
                var elem_idx = idx * simd_width
                var reduced_result = _load_reduce[
                    ngpus,
                    simd_width=simd_width,
                    alignment=alignment,
                    accum_type=accum_type,
                    use_multimem=use_multimem,
                ](elem_idx, ptrs)
                output_lambda[width=simd_width, alignment=alignment](
                    result.layout.idx2crd(elem_idx), reduced_result
                )
        else:
            var ptrs_ngpus = rebind[
                InlineArray[TileTensor[dtype, in_layout, ImmutAnyOrigin], ngpus]
            ](ptrs)
            var num_simd_vectors = num_elements // simd_width
            var simd_prefix_elems = num_simd_vectors * simd_width
            if num_simd_vectors > 0:
                for idx in range(global_tid, num_simd_vectors, total_threads):
                    var elem_idx = idx * simd_width
                    var reduced_result = _load_reduce[
                        ngpus,
                        simd_width=simd_width,
                        alignment=alignment,
                        accum_type=accum_type,
                        use_multimem=False,
                    ](elem_idx, ptrs_ngpus)
                    output_lambda[width=simd_width, alignment=alignment](
                        result.layout.idx2crd(elem_idx), reduced_result
                    )
            if simd_prefix_elems < num_elements:
                for elem_idx in range(
                    simd_prefix_elems + global_tid,
                    num_elements,
                    total_threads,
                ):
                    _allreduce_1stage_reduce_store_one[
                        dtype,
                        in_layout,
                        out_layout,
                        ngpus,
                        accum_type=accum_type,
                        output_lambda=output_lambda,
                    ](elem_idx, ptrs_ngpus, result)

        _multi_gpu_barrier[ngpus, is_start=False](rank_sigs, my_sig, my_rank)


@always_inline
def _allreduce_p2p[
    dtype: DType,
    ngpus: Int,
    in_layout: TensorLayout,
    in_origin: Origin,
    out_layout: TensorLayout,
    output_lambda: elementwise_epilogue_type,
    pdl_level: PDLLevel,
    use_multimem: Bool = False,
](
    list_of_in_tensors: InlineArray[
        TileTensor[dtype, in_layout, in_origin],
        1 if use_multimem else ngpus,
    ],
    out_tensor: TileTensor[mut=True, dtype, out_layout, ...],
    rank_sigs: InlineArray[UnsafePointer[Signal, MutAnyOrigin], MAX_GPUS],
    max_num_blocks: Int,
    ctx: DeviceContext,
) raises:
    """
    Performs allreduce using peer-to-peer access for a single GPU.

    Parameters:
        dtype: Data dtype of tensor elements.
        ngpus: Number of GPUs participating.
        in_layout: Layout of the input TileTensors.
        in_origin: Origin of the input TileTensors.
        out_layout: Layout of the output TileTensor.
        output_lambda: An output elementwise lambda.
        pdl_level: Control PDL behavior for the kernel.
        use_multimem: If True, use multi-memory space buffers for input.

    Args:
        list_of_in_tensors: Input buffers from ALL GPUs (peer access required)
        out_tensor: Output buffer for THIS GPU
        rank_sigs: Signal pointers for synchronization
        max_num_blocks: Maximum number of thread blocks to launch.
        ctx: Device context for THIS GPU

    Launches P2P reduction kernel on the current GPU to perform direct reduction.
    """
    comptime num_tensors = 1 if use_multimem else ngpus
    comptime simd_width = simd_width_of[dtype, target=get_gpu_target()]()
    var num_elements = list_of_in_tensors[0].num_elements()

    # Do nothing if there are no elements to reduce.
    if num_elements == 0:
        return

    if use_multimem and num_elements % simd_width != 0:
        raise Error(
            "multimem allreduce requires the element count to be a multiple of"
            " SIMD width"
        )

    # Flatten inputs to 1D - allreduce does not need dimension info
    comptime FlatLayout = type_of(row_major(Idx(num_elements)))
    comptime FlatIn = TileTensor[dtype, FlatLayout, ImmutAnyOrigin]
    var flat_inputs = InlineArray[FlatIn, num_tensors](uninitialized=True)
    comptime for i in range(num_tensors):
        flat_inputs[i] = FlatIn(
            rebind[UnsafePointer[Scalar[dtype], ImmutAnyOrigin]](
                list_of_in_tensors[i].ptr
            ),
            row_major(Idx(num_elements)),
        )

    # TODO(KERN-2632): Incorporate this into dispatch table
    comptime sm_version = ctx.default_device_info.version
    comptime BLOCK_SIZE = 512 if sm_version == "CDNA4" else 256

    comptime rank_4_byte_threshold = 512 * 1024
    comptime rank_8_byte_threshold = 256 * 1024
    var payload_bytecount = num_elements * size_of[dtype]()
    # The 2-stage path partitions by full SIMD vectors only; use 1-stage when a
    # scalar tail is present (unless multimem, which is rejected above).
    var latency_bound_small = (
        ngpus <= 4 and (payload_bytecount < rank_4_byte_threshold)
    ) or (ngpus <= 8 and (payload_bytecount < rank_8_byte_threshold))
    var use_1stage = latency_bound_small or (num_elements % simd_width != 0)

    if use_1stage:
        var grid_size: Int
        comptime if use_multimem:
            var simd_chunks = num_elements // simd_width
            var tail_elems_mm = num_elements - simd_chunks * simd_width
            grid_size = min(
                max_num_blocks,
                max(
                    1,
                    ceildiv(max(simd_chunks, tail_elems_mm), BLOCK_SIZE),
                ),
            )
        else:
            var num_simd_vecs = num_elements // simd_width
            var tail_elems = num_elements - num_simd_vecs * simd_width
            var grid_simd = ceildiv(num_simd_vecs, BLOCK_SIZE)
            var grid_tail = ceildiv(tail_elems, BLOCK_SIZE)
            grid_size = min(max_num_blocks, max(max(grid_simd, grid_tail), 1))

        # Use the 1-stage allreduce when transfer is latency bound.
        comptime allreduce_1stage_kernel = _allreduce_1stage_kernel[
            dtype,
            ngpus,
            FlatLayout,
            out_layout,
            BLOCK_SIZE=BLOCK_SIZE,
            output_lambda=output_lambda,
            use_multimem=use_multimem,
        ]
        ctx.enqueue_function[allreduce_1stage_kernel, allreduce_1stage_kernel](
            rebind[TileTensor[dtype, out_layout, MutAnyOrigin]](out_tensor),
            flat_inputs,
            rank_sigs,
            num_elements,
            Int(ctx.id()),
            grid_dim=grid_size,
            block_dim=BLOCK_SIZE,
            attributes=pdl_launch_attributes(pdl_level),
        )
    else:
        # Define grid size for 2-stage, which processes 1/ngpus of the
        # number of elements.
        var grid_size = min(
            max_num_blocks,
            ceildiv(num_elements // (simd_width * ngpus), BLOCK_SIZE),
        )

        # Otherwise, use 2-stage allreduce for the bandwidth bound regime.
        comptime kernel = _allreduce_2stage_kernel[
            dtype,
            ngpus,
            FlatLayout,
            out_layout,
            BLOCK_SIZE=BLOCK_SIZE,
            output_lambda=output_lambda,
            use_multimem=use_multimem,
        ]
        ctx.enqueue_function[kernel, kernel](
            rebind[TileTensor[dtype, out_layout, MutAnyOrigin]](out_tensor),
            flat_inputs,
            rank_sigs,
            num_elements,
            Int(ctx.id()),
            grid_dim=grid_size,
            block_dim=BLOCK_SIZE,
            attributes=pdl_launch_attributes(pdl_level),
        )


@parameter
def allreduce[
    dtype: DType,
    ngpus: Int,
    in_layout: TensorLayout,
    in_origin: Origin,
    out_layout: TensorLayout,
    output_lambda: Optional[elementwise_epilogue_type] = None,
    pdl_level: PDLLevel = PDLLevel(),
    *,
    use_multimem: Bool = False,
](
    input_tensors: InlineArray[
        TileTensor[dtype, in_layout, in_origin],
        1 if use_multimem else ngpus,
    ],
    output_tensor: TileTensor[mut=True, dtype, out_layout, ...],
    rank_sigs: InlineArray[UnsafePointer[Signal, MutAnyOrigin], MAX_GPUS],
    ctx: DeviceContext,
    _max_num_blocks: Optional[Int] = None,
) raises:
    """Per-device allreduce: one instance per GPU builds its own output.

    High-level model
    - Each GPU runs one instance of this function in parallel with the others.
    - Every instance reads all inputs but writes only its own output buffer.
    - A Python-level fence is inserted across the outputs to prevent reordering.

    Two execution paths
    1) P2P fast path (when peer access is available)
       - 1-stage kernel (latency-bound): each thread vector-loads from all GPUs,
         accumulates in higher precision, and writes directly to the result.
       - 2-stage kernel (bandwidth-bound): reduce-scatter then all-gather.
         Uses each GPU's `rank_sigs[*]` payload as a staging area for partitions.

         Diagram (per GPU r, 2-stage):
           - Stage 1: write reduced partition r into payload of `rank_sigs[r]`.
           - Stage 2: gather partitions from all peers' payloads into `out_r`.

    2) Naive fallback (no P2P)
       - For GPU r: create local accumulator A_r, allocate a temporary buffer S_r,
         copy each peer input into S_r and accumulate into A_r, then apply the epilogue
         into `out_r`.

         Diagram (per GPU r, naive):
           in_r -> A_r += in_r; for i!=r: in_i -> tmp_r -> A_r += tmp_r; A_r -> out_r

    Parameters:
        dtype: Data type of the tensor elements.
        ngpus: Number of GPUs participating in the allreduce.
        in_layout: Layout of the input TileTensors.
        in_origin: Origin of the input TileTensors.
        out_layout: Layout of the output TileTensor.
        output_lambda: Elementwise epilogue applied on the device result.
        pdl_level: Controls PDL behavior for P2P kernels.
        use_multimem: Whether to use multimem mode for improved performance.

    Args:
        input_tensors: Inputs from ALL GPUs as TileTensors.
        output_tensor: Output for THIS GPU as a TileTensor.
        rank_sigs: Per-GPU Signal pointers.
        ctx: Device context for THIS GPU.
        _max_num_blocks: Optional grid limit.

    Notes:
      - Inputs must have identical shape/dtype across GPUs.
      - Signal buffers must be sized at least `size_of(Signal) + payload_bytes`
        for the P2P 2-stage path, where `payload_bytes` equals the input
        tensor bytecount.
      - The naive path is automatically selected if P2P cannot be enabled.
      - The `use_multimem` parameter requires P2P access between GPUs.
    """
    comptime assert ngpus >= 2, "allreduce requires at least 2 GPUs"
    comptime num_tensors = 1 if use_multimem else ngpus

    # Return early, if the input buffer is empty
    var num_elements = input_tensors[0].num_elements()
    if num_elements == 0:
        return

    @always_inline
    @parameter
    @__copy_capture(output_tensor)
    def default_output_lambda[
        _dtype: DType,
        _width: SIMDSize,
        *,
        _alignment: Int,
    ](coords: Coord, val: SIMD[_dtype, _width]) -> None:
        output_tensor.store[width=_width, alignment=_alignment](
            coords, val.cast[dtype]()
        )

    comptime actual_output_lambda = default_output_lambda if not output_lambda else output_lambda.value()

    # TODO: check all devices have the same GPU sm_version
    comptime sm_version = ctx.default_device_info.version
    var num_bytes = num_elements * size_of[dtype]()
    var max_num_blocks = _max_num_blocks.or_else(
        dispatch_max_num_blocks[ngpus, sm_version, allreduce_tuning_table](
            num_bytes
        )
    )
    if max_num_blocks > MAX_NUM_BLOCKS_UPPER_BOUND:
        raise Error(
            "expected allreduce max_num_blocks less than upper bound: "
            + String(MAX_NUM_BLOCKS_UPPER_BOUND)
            + " but got: "
            + String(max_num_blocks)
        )

    # Check P2P availability.
    if not is_p2p_enabled():
        comptime if use_multimem:
            raise Error(
                "Allreduce with multimem requires P2P access between GPUs"
            )
        return _allreduce_naive_single[
            ngpus=ngpus,
            output_lambda=actual_output_lambda,
            num_tensors=1 if use_multimem else ngpus,
        ](input_tensors, output_tensor, max_num_blocks, ctx)

    # P2P path.
    return _allreduce_p2p[
        ngpus=ngpus,
        output_lambda=actual_output_lambda,
        pdl_level=pdl_level,
        use_multimem=use_multimem,
    ](input_tensors, output_tensor, rank_sigs, max_num_blocks, ctx)
