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


from std.math import ceildiv, iota
from std.sys.info import simd_width_of

from std.algorithm import elementwise
from std.bit import next_power_of_two
from std.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    barrier,
    block_idx_uint as block_idx,
    global_idx_uint as global_idx,
    thread_idx_uint as thread_idx,
)
import std.gpu.primitives.warp as warp
from std.gpu.host import DeviceContext, get_gpu_target
from std.gpu.host.info import is_cpu
from std.gpu.memory import AddressSpace
from std.memory import stack_allocation
from layout import Idx, TensorLayout, TileTensor, row_major
from std.runtime.tracing import Trace, TraceLevel, get_safe_task_id

from std.utils.index import IndexList, StaticTuple


def _argsort_cpu[
    *,
    ascending: Bool = True,
](
    indices: TileTensor[mut=True, address_space=AddressSpace.GENERIC, ...],
    input: TileTensor,
) raises:
    """
    Performs argsort on CPU.

    Parameters:
        ascending: Sort direction (True for ascending, False for descending).

    Args:
        indices: Output buffer to store sorted indices.
        input: Input buffer to sort.
    """
    comptime assert input.flat_rank == 1

    @parameter
    def fill_indices_iota[
        width: Int, rank: Int, alignment: Int = 1
    ](offset: IndexList[rank]):
        indices.ptr.store(
            offset[0],
            iota[indices.dtype, width](Scalar[indices.dtype](offset[0])),
        )

    elementwise[
        fill_indices_iota, simd_width_of[indices.dtype](), target="cpu"
    ](indices.num_elements())

    @parameter
    def cmp_fn(a: Scalar[indices.dtype], b: Scalar[indices.dtype]) -> Bool:
        comptime if ascending:
            return input[a] < input[b]
        else:
            return input[a] > input[b]

    sort[cmp_fn](
        Span[
            Scalar[indices.dtype],
            indices.origin,
        ](ptr=indices.ptr, length=indices.num_elements())
    )


@always_inline
def _sentinel_val[dtype: DType, ascending: Bool]() -> Scalar[dtype]:
    """
    Returns a sentinel value based on sort direction.

    Parameters:
        dtype: Data type of the sentinel value.
        ascending: Sort direction.

    Returns:
        MAX_FINITE for ascending sort, MIN_FINITE for descending sort.
    """

    comptime if ascending:
        return Scalar[dtype].MAX_FINITE
    else:
        return Scalar[dtype].MIN_FINITE


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](256))
def _bitonic_local_sort_kernel[
    input_dtype: DType,
    indices_dtype: DType,
    ascending: Bool,
    IndicesLayoutType: TensorLayout,
    InputLayoutType: TensorLayout,
](
    indices_arg: TileTensor[
        mut=True, indices_dtype, IndicesLayoutType, MutAnyOrigin
    ],
    input_arg: TileTensor[mut=True, input_dtype, InputLayoutType, MutAnyOrigin],
    n_arg: Int,
):
    """GPU kernel: local bitonic sort using shared memory.

    Each block independently sorts 256 elements. Fuses all stages from 1 to
    log2(256)=8 into a single kernel launch.
    """
    comptime BLOCK_SIZE = 256
    var tid = thread_idx.x
    var gid = Int(UInt(block_idx.x) * UInt(BLOCK_SIZE) + UInt(tid))
    var vals = input_arg.ptr
    var idxs = indices_arg.ptr

    var shared_vals = stack_allocation[
        BLOCK_SIZE,
        Scalar[input_dtype],
        address_space=AddressSpace.SHARED,
    ]()
    var shared_idxs = stack_allocation[
        BLOCK_SIZE,
        Scalar[indices_dtype],
        address_space=AddressSpace.SHARED,
    ]()

    if gid < n_arg:
        shared_vals[Int(tid)] = vals[gid]
        shared_idxs[Int(tid)] = idxs[gid]
    else:
        shared_vals[Int(tid)] = _sentinel_val[input_dtype, ascending]()
        shared_idxs[Int(tid)] = Scalar[indices_dtype](-1)

    var k = 2
    while k <= BLOCK_SIZE:
        var j = k >> 1
        while j > 0:
            barrier()
            var partner = UInt(tid) ^ UInt(j)
            if partner > UInt(tid):
                var vi = shared_vals[Int(tid)]
                var vp = shared_vals[Int(partner)]

                var cmp_val: Bool
                comptime if ascending:
                    cmp_val = vi > vp
                else:
                    cmp_val = vi < vp

                var direction = (UInt(tid) & UInt(k)) == 0
                if cmp_val == direction:
                    shared_vals[Int(tid)] = vp
                    shared_vals[Int(partner)] = vi
                    var ii = shared_idxs[Int(tid)]
                    shared_idxs[Int(tid)] = shared_idxs[Int(partner)]
                    shared_idxs[Int(partner)] = ii
            j >>= 1
        k <<= 1

    barrier()
    if gid < n_arg:
        vals[gid] = shared_vals[Int(tid)]
        idxs[gid] = shared_idxs[Int(tid)]


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](256))
def _bitonic_merge_local_kernel[
    input_dtype: DType,
    indices_dtype: DType,
    ascending: Bool,
    IndicesLayoutType: TensorLayout,
    InputLayoutType: TensorLayout,
](
    indices_arg: TileTensor[
        mut=True, indices_dtype, IndicesLayoutType, MutAnyOrigin
    ],
    input_arg: TileTensor[mut=True, input_dtype, InputLayoutType, MutAnyOrigin],
    n_arg: Int,
    stage: Int,
):
    """GPU kernel: fused local merge using shared memory.

    Fuses all steps < 256 within a global stage into a single kernel.
    Each block loads 256 contiguous elements, performs all local merge
    steps, then writes back.
    """
    comptime BLOCK_SIZE = 256
    var tid = thread_idx.x
    var gid = Int(UInt(block_idx.x) * UInt(BLOCK_SIZE) + UInt(tid))
    var vals = input_arg.ptr
    var idxs = indices_arg.ptr

    var shared_vals = stack_allocation[
        BLOCK_SIZE,
        Scalar[input_dtype],
        address_space=AddressSpace.SHARED,
    ]()
    var shared_idxs = stack_allocation[
        BLOCK_SIZE,
        Scalar[indices_dtype],
        address_space=AddressSpace.SHARED,
    ]()

    shared_vals[Int(tid)] = vals[gid]
    shared_idxs[Int(tid)] = idxs[gid]

    var j = BLOCK_SIZE >> 1
    while j > 0:
        barrier()
        var partner = UInt(tid) ^ UInt(j)
        if partner > UInt(tid):
            var vi = shared_vals[Int(tid)]
            var vp = shared_vals[Int(partner)]

            var cmp_val: Bool
            comptime if ascending:
                cmp_val = vi > vp
            else:
                cmp_val = vi < vp

            var direction = (UInt(gid) & UInt(stage)) == 0
            if cmp_val == direction:
                shared_vals[Int(tid)] = vp
                shared_vals[Int(partner)] = vi
                var ii = shared_idxs[Int(tid)]
                shared_idxs[Int(tid)] = shared_idxs[Int(partner)]
                shared_idxs[Int(partner)] = ii
        j >>= 1

    barrier()
    vals[gid] = shared_vals[Int(tid)]
    idxs[gid] = shared_idxs[Int(tid)]


def _argsort_gpu_impl[
    *,
    ascending: Bool = True,
](
    indices: TileTensor[mut=True, ...],
    input: TileTensor[mut=True, ...],
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
    def bitonic_global_step(
        indices_arg: TileTensor[
            indices.dtype, indices.LayoutType, indices.origin
        ],
        input_arg: TileTensor[input.dtype, input.LayoutType, input.origin],
        n_arg: Int,
        step: Int,
        stage: Int,
    ):
        var i = global_idx.x
        if i >= UInt(n_arg):
            return

        var partner = i ^ UInt(step)
        if partner > i and partner < UInt(n_arg):
            var cmp_val: Bool
            comptime if ascending:
                cmp_val = input_arg[i] > input_arg[partner]
            else:
                cmp_val = input_arg[i] < input_arg[partner]

            var bitonic_merge_direction = (i & UInt(stage)) == 0
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
    ]
    ctx.enqueue_function[local_sort_kernel, local_sort_kernel](
        indices,
        input,
        n,
        block_dim=BLOCK_SIZE,
        grid_dim=ceildiv(n, BLOCK_SIZE),
    )

    # Phase 2: Global merge stages (for stages beyond BLOCK_SIZE).
    var k = BLOCK_SIZE * 2
    while k <= n:
        # Global steps: step >= BLOCK_SIZE (partners in different blocks).
        var j = k // 2
        while j >= BLOCK_SIZE:
            comptime global_step_kernel = bitonic_global_step
            ctx.enqueue_function[global_step_kernel, global_step_kernel](
                indices,
                input,
                n,
                j,
                k,
                block_dim=BLOCK_SIZE,
                grid_dim=ceildiv(n, BLOCK_SIZE),
            )
            j //= 2

        # Fused local steps: step < BLOCK_SIZE, all within shared memory.
        comptime merge_local_kernel = _bitonic_merge_local_kernel[
            input_dtype=input.dtype,
            indices_dtype=indices.dtype,
            ascending=ascending,
            IndicesLayoutType=indices.LayoutType,
            InputLayoutType=input.LayoutType,
        ]
        ctx.enqueue_function[merge_local_kernel, merge_local_kernel](
            indices,
            input,
            n,
            k,
            block_dim=BLOCK_SIZE,
            grid_dim=ceildiv(n, BLOCK_SIZE),
        )
        k *= 2


def _argsort_gpu[
    *,
    ascending: Bool = True,
](
    indices: TileTensor[mut=True, ...],
    input: TileTensor[mut=True, ...],
    ctx: DeviceContext,
) raises:
    """
    Performs argsort on GPU with padding to power-of-two size if needed.

    Parameters:
        ascending: Sort direction (True for ascending, False for descending).

    Args:
        indices: Output buffer to store sorted indices.
        input: Input buffer to sort.
        ctx: Device context for GPU execution.
    """
    comptime assert indices.flat_rank == 1
    comptime assert input.flat_rank == 1
    # Create a device buffer to store a copy of the input data
    var n = indices.num_elements()

    if n.is_power_of_two():
        # Initialize indices with iota.
        @parameter
        @__copy_capture(indices)
        def fill_indices_iota_no_padding[
            width: Int, rank: Int, alignment: Int = 1
        ](offset: IndexList[rank]):
            indices.ptr.store(
                offset[0],
                iota[indices.dtype, width](Scalar[indices.dtype](offset[0])),
            )

        elementwise[
            fill_indices_iota_no_padding,
            simd_width=simd_width_of[indices.dtype, target=get_gpu_target()](),
            target="gpu",
        ](n, ctx)

        return _argsort_gpu_impl[ascending=ascending](indices, input, ctx)

    var pow_2_length = next_power_of_two(n)

    # Else we need to pad the input and indices with sentinel values.

    var padded_input_buffer = ctx.enqueue_create_buffer[input.dtype](
        pow_2_length
    )
    var padded_input = TileTensor(
        padded_input_buffer, row_major(Idx(pow_2_length))
    )

    var padded_indices_buffer = ctx.enqueue_create_buffer[indices.dtype](
        pow_2_length
    )
    var padded_indices = TileTensor(
        padded_indices_buffer,
        row_major(Idx(pow_2_length)),
    )

    # Initialize indices with sequential values and copy input data to device
    @parameter
    @__copy_capture(padded_indices, padded_input, input, indices, n)
    def fill_indices_iota[
        width: Int, rank: Int, alignment: Int = 1
    ](offset: IndexList[rank]):
        var i = offset[0]
        if i < n:
            padded_indices.ptr.store(
                i, iota[padded_indices.dtype, width](Scalar[indices.dtype](i))
            )
            padded_input.ptr.store[
                alignment=simd_width_of[padded_input.dtype]()
            ](i, input.ptr.load[width=width](i))
            return

        # otherwise we pad with a sentinel value and the max/min value for the type.
        comptime UNKNOWN_VALUE = -1
        padded_indices.ptr.store(
            i, SIMD[padded_indices.dtype, width](UNKNOWN_VALUE)
        )
        padded_input.ptr.store(
            i,
            SIMD[padded_input.dtype, width](
                _sentinel_val[padded_input.dtype, ascending]()
            ),
        )

    # we want to fill one element at a time to handle the case where n is not a
    # power of 2, so we set the simdwidth to be 1.
    elementwise[fill_indices_iota, simd_width=1, target="gpu"](
        pow_2_length, ctx
    )

    # Run the argsort implementation with the padded input and indices.
    _argsort_gpu_impl[ascending=ascending](padded_indices, padded_input, ctx)

    # Extract the unpadded indices from the padded indices.
    @parameter
    @__copy_capture(padded_indices, indices)
    def extract_indices[
        width: Int, rank: Int, alignment: Int = 1
    ](offset: IndexList[rank]):
        indices.ptr.store(
            offset[0], padded_indices.ptr.load[width=width](offset[0])
        )

    # Extract the unpadded indices from the padded indices.
    elementwise[
        extract_indices,
        simd_width=simd_width_of[indices.dtype, target=get_gpu_target()](),
        target="gpu",
    ](n, ctx)

    # Free the temporary input buffer
    _ = padded_input_buffer^
    _ = padded_indices_buffer^


def _validate_argsort(input: TileTensor, output: TileTensor) raises:
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
    target: StaticString = "cpu",
](
    output: TileTensor[mut=True, address_space=AddressSpace.GENERIC, ...],
    input: TileTensor[mut=True, ...],
    ctx: DeviceContext,
) raises:
    """
    Performs argsort on input buffer, storing indices in output buffer.

    Parameters:
        ascending: Sort direction (True for ascending, False for descending).
        target: Target device ("cpu" or "gpu").

    Args:
        output: Buffer to store sorted indices.
        input: Buffer containing values to sort.
        ctx: Device context for execution.
    """
    comptime assert input.flat_rank == 1
    comptime assert output.flat_rank == 1
    with Trace[TraceLevel.OP, target=target](
        "argsort",
        task_id=get_safe_task_id(ctx),
    ):
        _validate_argsort(input, output)

        comptime if is_cpu[target]():
            return _argsort_cpu[ascending=ascending](output, input)
        else:
            return _argsort_gpu[ascending=ascending](output, input, ctx)


def argsort[
    ascending: Bool = True
](
    output: TileTensor[mut=True, address_space=AddressSpace.GENERIC, ...],
    input: TileTensor,
) raises:
    """
    CPU-only version of argsort.

    Parameters:
        ascending: Sort direction (True for ascending, False for descending).

    Args:
        output: Buffer to store sorted indices.
        input: Buffer containing values to sort.
    """
    comptime assert input.flat_rank == 1
    comptime assert output.flat_rank == 1
    comptime assert output.dtype.is_integral()
    with Trace[TraceLevel.OP]("argsort"):
        _validate_argsort(input, output)
        _argsort_cpu[ascending=ascending](output, input)
