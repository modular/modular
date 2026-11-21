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
"""Provides CPU and GPU implementations of argsort that return indices permuting a tensor into sorted order."""


from std.math import ceildiv, iota
from std.sys.info import align_of, simd_width_of

from max.algorithm import elementwise
from std.bit import next_power_of_two
from max.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    block_idx,
    global_idx,
    thread_idx,
)
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext
from max.gpu.host.info import is_cpu
from max.gpu.primitives.grid_controls import (
    PDL,
    PDLLevel,
    pdl_launch_attributes,
)
from std.memory import unsafe_stack_allocation
from layout import (
    DefaultEngine,
    TensorLayout,
    TensorEngine,
    TileTensor,
    row_major,
)
from max.runtime.tracing import Trace, TraceLevel, get_safe_task_id

from std.utils.coord import Coord
from std.utils.index import StaticTuple


def _argsort_cpu[
    *,
    ascending: Bool = True,
    IndicesEngine: TensorEngine = DefaultEngine[element_width=1],
    InputEngine: TensorEngine = DefaultEngine[element_width=1],
](
    indices: TileTensor[
        mut=True, address_space=.GENERIC, Engine=IndicesEngine, ...
    ],
    input: TileTensor[mut=False, Engine=InputEngine, ...],
) raises:
    """
    Performs argsort on CPU.

    Parameters:
        ascending: Sort direction (True for ascending, False for descending).
        IndicesEngine: Engine of the indices tile.
        InputEngine: Engine of the input tile.

    Args:
        indices: Output buffer to store sorted indices.
        input: Input buffer to sort.
    """
    comptime assert input.flat_rank == 1

    def fill_indices_iota[width: Int, alignment: Int = 1](offset: Coord) {var}:
        indices.store(
            offset,
            iota[indices.dtype, width](
                Scalar[indices.dtype](offset[0].value())
            ),
        )

    elementwise[simd_width_of[indices.dtype]()](
        fill_indices_iota,
        Coord(indices.num_elements()),
        DeviceContext.cpu(),
    )

    def cmp_fn(
        a: Scalar[indices.dtype], b: Scalar[indices.dtype]
    ) {input} -> Bool:
        comptime assert a.dtype.is_integral()
        comptime assert b.dtype.is_integral()
        comptime if ascending:
            return input[a] < input[b]
        else:
            return input[a] > input[b]

    sort(indices.as_span(), cmp_fn)


@inline(.always)
def _sentinel_val[dtype: DType, ascending: Bool]() -> Scalar[dtype]:
    """
    Returns a sentinel value based on sort direction.

    Parameters:
        dtype: Data type of the sentinel value.
        ascending: Sort direction.

    Returns:
        MAX for ascending sort, MIN for descending sort (+inf / -inf for
        floating-point types). No real value sorts beyond the sentinel; a
        real value equal to it is ordered first by `_sorts_after`.
    """

    comptime if ascending:
        return Scalar[dtype].MAX
    else:
        return Scalar[dtype].MIN


comptime _PAD_INDEX = -1
"""Index stored in the padding slots of the GPU sort buffers."""


@inline(.always)
def _is_pad[dtype: DType](idx: Scalar[dtype]) -> Bool:
    """Returns True if `idx` marks a padding slot."""
    return idx == Scalar[dtype](_PAD_INDEX)


@inline(.always)
def _sorts_after[
    ascending: Bool, dtype: DType
](va: Scalar[dtype], vb: Scalar[dtype], a_is_pad: Bool, b_is_pad: Bool) -> Bool:
    """Returns True if element a belongs after element b in sorted order.

    Padding holds the sentinel (the extreme value in the sort direction), so
    it can only tie with a real element, never sort before one by value. On a
    tie, padding goes last. This keeps every real element, including +-inf
    and values equal to the sentinel, in the first `n` slots.

    Parameters:
        ascending: Sort direction.
        dtype: Data type of the values.

    Args:
        va: Value of element a.
        vb: Value of element b.
        a_is_pad: Whether element a is padding.
        b_is_pad: Whether element b is padding.

    Returns:
        True if element a must be placed after element b.
    """
    if va == vb:
        return a_is_pad and not b_is_pad
    comptime if ascending:
        return va > vb
    else:
        return va < vb


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](256))
@__name(t"bitonic_local_sort_{input_dtype}_{indices_dtype}_{ascending}")
def _bitonic_local_sort_kernel[
    input_dtype: DType,
    indices_dtype: DType,
    ascending: Bool,
    IndicesLayoutType: TensorLayout,
    InputLayoutType: TensorLayout,
    IndicesEngine: TensorEngine = DefaultEngine[element_width=1],
    InputEngine: TensorEngine = DefaultEngine[element_width=1],
](
    indices_arg: TileTensor[
        mut=True,
        indices_dtype,
        IndicesLayoutType,
        MutAnyOrigin,
        Engine=IndicesEngine,
    ],
    input_arg: TileTensor[
        mut=True,
        input_dtype,
        InputLayoutType,
        MutAnyOrigin,
        Engine=InputEngine,
    ],
    n_arg: Int32,
):
    """GPU kernel: local bitonic sort using shared memory.

    Each block sorts 256 elements, with adjacent blocks finishing in opposite
    directions. Fuses all stages from 1 to log2(256)=8 into one kernel launch.
    """
    var _n_arg = Int(n_arg)
    comptime BLOCK_SIZE = 256
    var tid = thread_idx.x
    var gid: Int = block_idx.x * BLOCK_SIZE + tid
    var shared_vals = unsafe_stack_allocation[
        BLOCK_SIZE,
        Scalar[input_dtype],
        address_space=.SHARED,
    ]()
    var shared_idxs = unsafe_stack_allocation[
        BLOCK_SIZE,
        Scalar[indices_dtype],
        address_space=.SHARED,
    ]()

    with PDL():
        if gid < _n_arg:
            shared_vals.unsafe_store(tid, input_arg.load[width=1]((gid,)))
            shared_idxs.unsafe_store(tid, indices_arg.load[width=1]((gid,)))
        else:
            shared_vals.unsafe_store(
                tid, _sentinel_val[input_dtype, ascending]()
            )
            shared_idxs.unsafe_store(tid, Scalar[indices_dtype](_PAD_INDEX))

        var k = 2
        while k <= BLOCK_SIZE:
            var j = k >> 1
            while j > 0:
                barrier()
                var partner = tid ^ j
                if partner > tid:
                    var vi = shared_vals.unsafe_load(tid)
                    var vp = shared_vals.unsafe_load(partner)
                    var ii = shared_idxs.unsafe_load(tid)
                    var ip = shared_idxs.unsafe_load(partner)

                    var cmp_val = _sorts_after[ascending](
                        vi, vp, _is_pad(ii), _is_pad(ip)
                    )

                    # Global merges require adjacent blocks to form bitonic runs.
                    var direction = (gid & k) == 0
                    if cmp_val == direction:
                        shared_vals.unsafe_store(tid, vp)
                        shared_vals.unsafe_store(partner, vi)
                        shared_idxs.unsafe_store(tid, ip)
                        shared_idxs.unsafe_store(partner, ii)
                j >>= 1
            k <<= 1

        barrier()
        if gid < _n_arg:
            input_arg.store((gid,), shared_vals.unsafe_load(tid))
            indices_arg.store((gid,), shared_idxs.unsafe_load(tid))


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](256))
@__name(
    t"bitonic_merge_local_{input_dtype}_{indices_dtype}_{ascending}",
)
def _bitonic_merge_local_kernel[
    input_dtype: DType,
    indices_dtype: DType,
    ascending: Bool,
    IndicesLayoutType: TensorLayout,
    InputLayoutType: TensorLayout,
    IndicesEngine: TensorEngine = DefaultEngine[element_width=1],
    InputEngine: TensorEngine = DefaultEngine[element_width=1],
](
    indices_arg: TileTensor[
        mut=True,
        indices_dtype,
        IndicesLayoutType,
        MutAnyOrigin,
        Engine=IndicesEngine,
    ],
    input_arg: TileTensor[
        mut=True,
        input_dtype,
        InputLayoutType,
        MutAnyOrigin,
        Engine=InputEngine,
    ],
    n_arg: Int32,
    stage: Int32,
):
    """GPU kernel: fused local merge using shared memory.

    Fuses all steps < 256 within a global stage into a single kernel.
    Each block loads 256 contiguous elements, performs all local merge
    steps, then writes back.
    """
    var _stage = Int(stage)
    comptime BLOCK_SIZE = 256
    var tid = thread_idx.x
    var gid: Int = block_idx.x * BLOCK_SIZE + tid
    var shared_vals = unsafe_stack_allocation[
        BLOCK_SIZE,
        Scalar[input_dtype],
        address_space=.SHARED,
    ]()
    var shared_idxs = unsafe_stack_allocation[
        BLOCK_SIZE,
        Scalar[indices_dtype],
        address_space=.SHARED,
    ]()

    with PDL():
        shared_vals.unsafe_store(tid, input_arg.load[width=1]((gid,)))
        shared_idxs.unsafe_store(tid, indices_arg.load[width=1]((gid,)))

        var j = BLOCK_SIZE >> 1
        while j > 0:
            barrier()
            var partner = tid ^ j
            if partner > tid:
                var vi = shared_vals.unsafe_load(tid)
                var vp = shared_vals.unsafe_load(partner)
                var ii = shared_idxs.unsafe_load(tid)
                var ip = shared_idxs.unsafe_load(partner)

                var cmp_val = _sorts_after[ascending](
                    vi, vp, _is_pad(ii), _is_pad(ip)
                )

                var direction = (gid & _stage) == 0
                if cmp_val == direction:
                    shared_vals.unsafe_store(tid, vp)
                    shared_vals.unsafe_store(partner, vi)
                    shared_idxs.unsafe_store(tid, ip)
                    shared_idxs.unsafe_store(partner, ii)
            j >>= 1

        barrier()
        input_arg.store((gid,), shared_vals.unsafe_load(tid))
        indices_arg.store((gid,), shared_idxs.unsafe_load(tid))


def _argsort_gpu_impl[
    *,
    ascending: Bool = True,
    IndicesEngine: TensorEngine = DefaultEngine[element_width=1],
    InputEngine: TensorEngine = DefaultEngine[element_width=1],
](
    indices: TileTensor[mut=True, Engine=IndicesEngine, ...],
    input: TileTensor[mut=True, Engine=InputEngine, ...],
    ctx: DeviceContext,
) raises:
    """
    Implements GPU argsort using an optimized bitonic sort algorithm.

    Uses three kernels to minimize kernel launches and maximize data
    reuse through shared memory:
    1. Local sort: each block sorts BLOCK_SIZE elements entirely in shared
       memory, fusing all stages up to log2(BLOCK_SIZE) into one launch.
    2. Global merge step: for steps >= BLOCK_SIZE where partners are in
       different blocks.
    3. Fused local merge: fuses all steps < BLOCK_SIZE within a global
       stage into a single kernel using shared memory.

    Parameters:
        ascending: Sort direction (True for ascending, False for descending).
        IndicesEngine: Engine of the indices tile.
        InputEngine: Engine of the input tile.

    Args:
        indices: Output buffer to store sorted indices.
        input: Input buffer to sort.
        ctx: Device context for GPU execution.
    """
    comptime assert input.flat_rank == 1
    comptime assert indices.flat_rank == 1
    var n = indices.num_elements()

    assert n.is_power_of_two(), "n must be a power of two"

    comptime BLOCK_SIZE = 256

    # Global merge step kernel (nested: simple enough, no shared memory).
    @__llvm_metadata(
        MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](BLOCK_SIZE)
    )
    @__name(t"bitonic_global_step_{ascending}")
    def bitonic_global_step(
        indices_arg: TileTensor[
            indices.dtype,
            indices.LayoutType,
            indices.origin,
            Engine=indices.Engine,
        ],
        input_arg: TileTensor[
            input.dtype,
            input.LayoutType,
            input.origin,
            Engine=input.Engine,
        ],
        n_arg: Int32,
        step: Int32,
        stage: Int32,
    ):
        var _n_arg = Int(n_arg)
        var _step = Int(step)
        var _stage = Int(stage)
        var i = global_idx.x
        if i >= _n_arg:
            return

        var partner = i ^ _step
        with PDL():
            if partner > i and partner < _n_arg:
                var vi = input_arg.load[width=1]((i,))
                var vp = input_arg.load[width=1]((partner,))
                var cmp_val: Bool
                if vi == vp:
                    # Only ties need the indices (padding loses ties).
                    cmp_val = _sorts_after[ascending](
                        vi,
                        vp,
                        _is_pad(indices_arg.load[width=1]((i,))),
                        _is_pad(indices_arg.load[width=1]((partner,))),
                    )
                else:
                    cmp_val = _sorts_after[ascending](vi, vp, False, False)

                var bitonic_merge_direction = (i & _stage) == 0
                if cmp_val == bitonic_merge_direction:
                    swap(input_arg[i], input_arg[partner])
                    swap(indices_arg[i], indices_arg[partner])

    # ---- Main orchestration ----

    # Phase 1: Local sort - each block independently sorts BLOCK_SIZE
    # elements in shared memory.
    comptime local_sort_kernel = _bitonic_local_sort_kernel[
        input_dtype=input.dtype,
        indices_dtype=indices.dtype,
        ascending=ascending,
        IndicesLayoutType=indices.LayoutType,
        InputLayoutType=input.LayoutType,
        IndicesEngine=IndicesEngine,
        InputEngine=InputEngine,
    ]
    ctx.enqueue_function[local_sort_kernel](
        indices,
        input,
        Int32(n),
        block_dim=BLOCK_SIZE,
        grid_dim=ceildiv(n, BLOCK_SIZE),
        attributes=pdl_launch_attributes(PDLLevel.ON),
    )

    # Phase 2: Global merge stages (for stages beyond BLOCK_SIZE).
    var k = BLOCK_SIZE * 2
    while k <= n:
        # Global steps: step >= BLOCK_SIZE (partners in different blocks).
        var j = k // 2
        while j >= BLOCK_SIZE:
            comptime global_step_kernel = bitonic_global_step
            ctx.enqueue_function[global_step_kernel](
                indices,
                input,
                Int32(n),
                Int32(j),
                Int32(k),
                block_dim=BLOCK_SIZE,
                grid_dim=ceildiv(n, BLOCK_SIZE),
                attributes=pdl_launch_attributes(PDLLevel.ON),
            )
            j //= 2

        # Fused local steps: step < BLOCK_SIZE, all within shared memory.
        comptime merge_local_kernel = _bitonic_merge_local_kernel[
            input_dtype=input.dtype,
            indices_dtype=indices.dtype,
            ascending=ascending,
            IndicesLayoutType=indices.LayoutType,
            InputLayoutType=input.LayoutType,
            IndicesEngine=IndicesEngine,
            InputEngine=InputEngine,
        ]
        ctx.enqueue_function[merge_local_kernel](
            indices,
            input,
            Int32(n),
            Int32(k),
            block_dim=BLOCK_SIZE,
            grid_dim=ceildiv(n, BLOCK_SIZE),
            attributes=pdl_launch_attributes(PDLLevel.ON),
        )
        k *= 2


def _argsort_gpu[
    *,
    ascending: Bool = True,
    IndicesEngine: TensorEngine = DefaultEngine[element_width=1],
    InputEngine: TensorEngine = DefaultEngine[element_width=1],
](
    indices: TileTensor[mut=True, Engine=IndicesEngine, ...],
    input: TileTensor[mut=False, Engine=InputEngine, ...],
    ctx: DeviceContext,
) raises:
    """
    Performs argsort on GPU with padding to power-of-two size if needed.

    Parameters:
        ascending: Sort direction (True for ascending, False for descending).
        IndicesEngine: Engine of the indices tile.
        InputEngine: Engine of the input tile.

    Args:
        indices: Output buffer to store sorted indices.
        input: Input buffer to sort.
        ctx: Device context for GPU execution.
    """
    comptime assert indices.flat_rank == 1
    comptime assert input.flat_rank == 1
    # Create a device buffer to store a copy of the input data
    var n = indices.num_elements()
    var input_flat = input.reshape(row_major(input.num_elements()))
    var indices_flat = indices.reshape(row_major(n))

    if n.is_power_of_two():
        var input_copy_buffer = ctx.enqueue_create_buffer[input.dtype](n)
        var input_copy = TileTensor(input_copy_buffer, row_major(n))

        # Initialize indices with iota.
        def fill_indices_iota_no_padding[
            width: Int, alignment: Int = 1
        ](offset: Coord) {var}:
            var i = offset[0].value()

            indices_flat.store[alignment=align_of[indices.dtype]()](
                (i,),
                iota[indices.dtype, width](Scalar[indices.dtype](i)),
            )
            input_copy.store[alignment=simd_width_of[input_copy.dtype]()](
                (i,),
                input_flat.load[width=width, alignment=align_of[input.dtype]()](
                    (i,)
                ),
            )

        elementwise[
            simd_width=min(
                simd_width_of[indices.dtype, target=ctx.target](),
                simd_width_of[input.dtype, target=ctx.target](),
            ),
            _trace_description="argsort_fill_indices",
        ](fill_indices_iota_no_padding, Coord(n), ctx)

        _argsort_gpu_impl[ascending=ascending](indices, input_copy, ctx)
        _ = input_copy_buffer^
        return

    var pow_2_length = next_power_of_two(n)

    # Else we need to pad the input and indices with sentinel values.

    var padded_input_buffer = ctx.enqueue_create_buffer[input.dtype](
        pow_2_length
    )
    var padded_input = TileTensor(padded_input_buffer, row_major(pow_2_length))

    var padded_indices_buffer = ctx.enqueue_create_buffer[indices.dtype](
        pow_2_length
    )
    var padded_indices = TileTensor(
        padded_indices_buffer,
        row_major(pow_2_length),
    )

    # Initialize indices with sequential values and copy input data to device
    def fill_indices_iota[width: Int, alignment: Int = 1](offset: Coord) {var}:
        var i = Int(offset[0])
        if i < n:
            padded_indices.store[alignment=align_of[padded_indices.dtype]()](
                (i,),
                iota[padded_indices.dtype, width](Scalar[indices.dtype](i)),
            )
            padded_input.store[alignment=simd_width_of[padded_input.dtype]()](
                (i,),
                input_flat.load[width=width, alignment=align_of[input.dtype]()](
                    (i,)
                ),
            )
            return

        # Otherwise pad with index -1 and the sentinel value. `_sorts_after`
        # keeps padding behind every real element.
        padded_indices.store[alignment=align_of[padded_indices.dtype]()](
            (i,), SIMD[padded_indices.dtype, width](_PAD_INDEX)
        )
        padded_input.store[alignment=align_of[padded_input.dtype]()](
            (i,),
            SIMD[padded_input.dtype, width](
                _sentinel_val[padded_input.dtype, ascending]()
            ),
        )

    # we want to fill one element at a time to handle the case where n is not a
    # power of 2, so we set the simdwidth to be 1.
    elementwise[
        simd_width=1,
        _trace_description="argsort_fill_indices_padded",
    ](fill_indices_iota, Coord(pow_2_length), ctx)

    # Run the argsort implementation with the padded input and indices.
    _argsort_gpu_impl[ascending=ascending](padded_indices, padded_input, ctx)

    # Extract the unpadded indices from the padded indices.
    def extract_indices[width: Int, alignment: Int = 1](offset: Coord) {var}:
        indices.store(offset, padded_indices.load[width=width](offset))

    # Extract the unpadded indices from the padded indices.
    elementwise[
        simd_width=simd_width_of[indices.dtype, target=ctx.target](),
        _trace_description="argsort_extract_indices",
    ](extract_indices, Coord(n), ctx)

    # Free the temporary input buffer
    _ = padded_input_buffer^
    _ = padded_indices_buffer^


def _validate_argsort[
    IndicesEngine: TensorEngine = DefaultEngine[element_width=1],
    InputEngine: TensorEngine = DefaultEngine[element_width=1],
](
    input: TileTensor[mut=False, Engine=InputEngine, ...],
    output: TileTensor[mut=False, Engine=IndicesEngine, ...],
) raises:
    """
    Validates input and output buffers for argsort operation.

    Args:
        input: Buffer containing values to sort.
        output: Buffer to store sorted indices.

    Raises:
        Error if buffers don't meet requirements for argsort.
    """

    if output.num_elements() != input.num_elements():
        raise "output and input must have the same length"


def argsort[
    *,
    ascending: Bool = True,
    IndicesEngine: TensorEngine = DefaultEngine[element_width=1],
    InputEngine: TensorEngine = DefaultEngine[element_width=1],
](
    output: TileTensor[
        mut=True, address_space=.GENERIC, Engine=IndicesEngine, ...
    ],
    input: TileTensor[mut=False, Engine=InputEngine, ...],
    ctx: DeviceContext,
) raises:
    """
    Performs argsort on input buffer, storing indices in output buffer.

    Parameters:
        ascending: Sort direction (True for ascending, False for descending).
        IndicesEngine: Engine of the output (indices) tile.
        InputEngine: Engine of the input tile.

    Args:
        output: Buffer to store sorted indices.
        input: Buffer containing values to sort.
        ctx: Device context for execution.
    """
    comptime assert input.flat_rank == 1
    comptime assert output.flat_rank == 1
    with Trace[TraceLevel.OP, target=ctx.target.device_kind_label()](
        "argsort",
        task_id=get_safe_task_id(ctx),
    ):
        _validate_argsort[IndicesEngine, InputEngine](input, output)

        comptime if ctx.target.is_cpu():
            return _argsort_cpu[ascending=ascending](output, input)
        else:
            return _argsort_gpu[ascending=ascending](output, input, ctx)


def argsort[
    ascending: Bool = True,
    IndicesEngine: TensorEngine = DefaultEngine[element_width=1],
    InputEngine: TensorEngine = DefaultEngine[element_width=1],
](
    output: TileTensor[
        mut=True, address_space=.GENERIC, Engine=IndicesEngine, ...
    ],
    input: TileTensor[mut=False, Engine=InputEngine, ...],
) raises:
    """
    CPU-only version of argsort.

    Parameters:
        ascending: Sort direction (True for ascending, False for descending).
        IndicesEngine: Engine of the output (indices) tile.
        InputEngine: Engine of the input tile.

    Args:
        output: Buffer to store sorted indices.
        input: Buffer containing values to sort.
    """
    comptime assert input.flat_rank == 1
    comptime assert output.flat_rank == 1
    comptime assert output.dtype.is_integral()
    with Trace[TraceLevel.OP]("argsort"):
        _validate_argsort[IndicesEngine, InputEngine](input, output)
        _argsort_cpu[ascending=ascending](output, input)
