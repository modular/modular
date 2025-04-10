# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


from sys.info import simdwidthof, bitwidthof, alignof
from utils.index import Index
from buffer import NDBuffer
from gpu.host import DeviceContext
from collections.string import StaticString
from gpu.host.info import is_cpu
from math import iota, ceildiv
from algorithm import elementwise
from utils.index import IndexList, StaticTuple
from gpu.host._compile import _get_gpu_target
from gpu import MAX_THREADS_PER_BLOCK_METADATA, global_idx
from runtime.tracing import Trace, TraceLevel
from math import next_power_of_two


fn _argsort_cpu[
    *,
    ascending: Bool = True,
](indices: NDBuffer[mut=True, *_], input: NDBuffer) raises:
    """
    Performs argsort on CPU.

    Parameters:
        ascending: Sort direction (True for ascending, False for descending).

    Args:
        indices: Output buffer to store sorted indices.
        input: Input buffer to sort.
    """

    @parameter
    fn fill_indices_iota[width: Int, rank: Int](offset: IndexList[rank]):
        indices.store(offset[0], iota[indices.type, width](offset[0]))

    elementwise[fill_indices_iota, simdwidthof[indices.type](), target="cpu"](
        len(indices)
    )

    @parameter
    fn cmp_fn(a: Scalar[indices.type], b: Scalar[indices.type]) -> Bool:
        @parameter
        if ascending:
            return input[Int(a)] < input[Int(b)]
        else:
            return input[Int(a)] > input[Int(b)]

    sort[cmp_fn](
        Span[Scalar[indices.type], indices.origin](
            ptr=indices.data, length=len(indices)
        )
    )


@always_inline
fn _sentinel_val[type: DType, ascending: Bool]() -> Scalar[type]:
    """
    Returns a sentinel value based on sort direction.

    Parameters:
        type: Data type of the sentinel value.
        ascending: Sort direction.

    Returns:
        MAX_FINITE for ascending sort, MIN_FINITE for descending sort.
    """

    @parameter
    if ascending:
        return Scalar[type].MAX_FINITE
    else:
        return Scalar[type].MIN_FINITE


fn _argsort_gpu_impl[
    *,
    ascending: Bool = True,
](indices: NDBuffer[mut=True, *_], input: NDBuffer, ctx: DeviceContext,) raises:
    """
    Implements GPU argsort using bitonic sort algorithm.

    Parameters:
        ascending: Sort direction (True for ascending, False for descending).

    Args:
        indices: Output buffer to store sorted indices.
        input: Input buffer to sort.
        ctx: Device context for GPU execution.
    """
    # Create a device buffer to store a copy of the input data
    var n = len(indices)

    debug_assert(n.is_power_of_two(), "n must be a power of two")

    # Define block size for GPU kernel execution
    alias BLOCK_SIZE = 256

    # Bitonic sort algorithm implementation
    @__llvm_metadata(
        MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](BLOCK_SIZE)
    )
    fn bitonic_sort_step(
        indices: NDBuffer[mut=True, indices.type, indices.rank],
        input: NDBuffer[mut=True, input.type, input.rank],
        n: Int,
        step: Int,
        stage: Int,
    ):
        """
        GPU kernel for one step of the bitonic sort algorithm.

        Args:
            indices: Buffer containing indices to sort.
            input: Buffer containing values to sort by.
            n: Total length of the buffers (padded).
            step: Current step size in the bitonic sequence.
            stage: Current stage of the bitonic sort.
        """
        var i = global_idx.x

        if i >= n:
            return

        var partner = i ^ step

        if partner > i and partner < n:
            var cmp_val: Bool

            @parameter
            if ascending:
                cmp_val = input[i] > input[partner]
            else:
                cmp_val = input[i] < input[partner]

            # Determine if we are in ascending or descending part of bitonic merge.
            var bitonic_merge_direction = (i & stage) == 0

            if cmp_val == bitonic_merge_direction:
                swap(input[i], input[partner])
                swap(indices[i], indices[partner])

    var k = 2
    # Iterate through increasing sequence lengths
    while k <= n:
        var j = k // 2
        while j > 0:
            # Launch GPU kernel for each stage of the bitonic sort
            ctx.enqueue_function[bitonic_sort_step](
                indices,
                input,
                n,
                j,
                k,
                block_dim=BLOCK_SIZE,
                grid_dim=ceildiv(n, BLOCK_SIZE),
            )
            j //= 2
        k *= 2


fn _argsort_gpu[
    *,
    ascending: Bool = True,
](indices: NDBuffer[mut=True, *_], input: NDBuffer, ctx: DeviceContext) raises:
    """
    Performs argsort on GPU with padding to power-of-two size if needed.

    Parameters:
        ascending: Sort direction (True for ascending, False for descending).

    Args:
        indices: Output buffer to store sorted indices.
        input: Input buffer to sort.
        ctx: Device context for GPU execution.
    """
    # Create a device buffer to store a copy of the input data
    var n = len(indices)

    if n.is_power_of_two():
        # Initialize indices with iota.
        @parameter
        @__copy_capture(indices)
        fn fill_indices_iota_no_padding[
            width: Int, rank: Int
        ](offset: IndexList[rank]):
            indices.store(offset[0], iota[indices.type, width](offset[0]))

        elementwise[
            fill_indices_iota_no_padding,
            simd_width = simdwidthof[
                indices.type, target = _get_gpu_target()
            ](),
            target="gpu",
        ](n, ctx)

        return _argsort_gpu_impl[ascending=ascending](indices, input, ctx)

    var pow_2_length = next_power_of_two(n)

    # Else we need to pad the input and indices with sentinel values.

    var padded_input_buffer = ctx.enqueue_create_buffer[input.type](
        pow_2_length
    )
    var padded_input = NDBuffer[
        mut=True, input.type, input.rank, indices.origin
    ](padded_input_buffer._unsafe_ptr(), (pow_2_length))

    var padded_indices_buffer = ctx.enqueue_create_buffer[indices.type](
        pow_2_length
    )
    var padded_indices = NDBuffer[
        mut=True, indices.type, indices.rank, indices.origin
    ](padded_indices_buffer._unsafe_ptr(), (pow_2_length))

    # Initialize indices with sequential values and copy input data to device
    @parameter
    @__copy_capture(padded_indices, padded_input, input, indices, n)
    fn fill_indices_iota[width: Int, rank: Int](offset: IndexList[rank]):
        var i = offset[0]
        if i < n:
            padded_indices.store(i, iota[padded_indices.type, width](i))
            padded_input.store[width=width, alignment = padded_input.alignment](
                i, input.load[width=width](i)
            )
            return

        # otherwise we pad with a sentinel value and the max/min value for the type.
        padded_indices.store[width=width](i, -1)
        padded_input.store[width=width](
            i, _sentinel_val[padded_input.type, ascending]()
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
    fn extract_indices[width: Int, rank: Int](offset: IndexList[rank]):
        indices.store(offset[0], padded_indices.load[width=width](offset[0]))

    # Extract the unpadded indices from the padded indices.
    elementwise[
        extract_indices,
        simd_width = simdwidthof[indices.type, target = _get_gpu_target()](),
        target="gpu",
    ](n, ctx)

    # Free the temporary input buffer
    _ = padded_input_buffer^
    _ = padded_indices_buffer^


fn _validate_argsort(output: NDBuffer, input: NDBuffer) raises:
    """
    Validates input and output buffers for argsort operation.

    Args:
        output: Buffer to store sorted indices.
        input: Buffer containing values to sort.

    Raises:
        Error if buffers don't meet requirements for argsort.
    """

    @parameter
    if output.rank != 1:
        raise "output must be a 1D tensor"

    @parameter
    if input.rank != 1:
        raise "input must be a 1D tensor"

    @parameter
    if not output.type.is_integral():
        raise "output must be an integer type"

    if len(output) != len(input):
        raise "output and input must have the same length"


fn argsort[
    *,
    ascending: Bool = True,
    target: StaticString = "cpu",
](output: NDBuffer[mut=True, *_], input: NDBuffer, ctx: DeviceContext) raises:
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
    with Trace[TraceLevel.OP, target=target]("argsort"):
        _validate_argsort(output, input)

        @parameter
        if is_cpu[target]():
            return _argsort_cpu[ascending=ascending](output, input)
        else:
            return _argsort_gpu[ascending=ascending](output, input, ctx)


fn argsort[
    ascending: Bool = True
](output: NDBuffer[mut=True, *_], input: NDBuffer) raises:
    """
    CPU-only version of argsort.

    Parameters:
        ascending: Sort direction (True for ascending, False for descending).

    Args:
        output: Buffer to store sorted indices.
        input: Buffer containing values to sort.
    """
    with Trace[TraceLevel.OP]("argsort"):
        _validate_argsort(output, input)
        _argsort_cpu[ascending=ascending](output, input)
