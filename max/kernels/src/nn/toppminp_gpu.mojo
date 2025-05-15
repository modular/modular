# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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


from collections import OptionalReg
from math import ceildiv, exp
from sys import alignof, bitwidthof, simdwidthof, sizeof
from sys._assembly import inlined_assembly

from buffer import NDBuffer
from buffer.dimlist import DimList
from builtin.dtype import _uint_type_of_width
from builtin.io import _printf
from gpu import (
    WARP_SIZE,
    barrier,
    block_dim,
    block_idx,
    grid_dim,
    lane_id,
    thread_idx,
)
from gpu.host import DeviceContext
from gpu.host.dim import Dim
from gpu.memory import AddressSpace, external_memory
from gpu.random import Random
from memory import UnsafePointer, bitcast, stack_allocation
from nn.softmax import _softmax_gpu
from nn.topk import (
    TopK_2,
    _block_reduce_topk,
    _get_shmem_size_stg_1,
    _topk_dead_val,
)

from utils import IndexList

alias DEBUG_FILE = False
alias SEED = 42


fn topk_wrapper[
    T: DType,
    out_idx_type: DType,
    is_top_p: Bool,
    largest: Bool = True,
    _test_sort: Bool = False,
](
    K: Int,
    num_elements: Int,
    num_blocks_per_input: Int,
    in_buffer: UnsafePointer[Scalar[T]],
    local_topk_vals: UnsafePointer[
        Scalar[T]
    ],  # Output buffer of size num_blocks_per_input * K
    local_topk_idxs: UnsafePointer[
        Scalar[out_idx_type]
    ],  # Output buffer of size num_blocks_per_input * K
    p_threshold: UnsafePointer[Scalar[T]],
    skip_sort: UnsafePointer[Scalar[DType.bool]],
):
    """
    Copy of `Kernels/mojo/nn/topk.mojo:_topk_stage1` with the addition of
    max_vals and p_threshold arguments to determine if sorting is needed for
    top-p/min-p sampling.

    Parameters:
        T: DType - The data type of the elements.
        out_idx_type: DType - The data type of the output indices.
        is_top_p: Bool - Whether this if for top-p sampling or min-p sampling.
        largest: Bool - Whether to find the maximum or minimum value.
        _test_sort: Bool - An internal test flag to not skip sort if testing.

    Arguments:
        K: Int - Number of top elements to select per block
        num_elements: Int - Size of last dimension of input buffer (vocab size)
        num_blocks_per_input: Int - Number of blocks used to process the input data
        in_buffer: UnsafePointer[Scalar[T]] - Input buffer containing the elements to process
        local_topk_vals: UnsafePointer[Scalar[T]] - Output buffer to store the local top-K values
        local_topk_idxs: UnsafePointer[Scalar[out_idx_type]] - Output buffer to store the indices of local top-K elements
        p_threshold: UnsafePointer[Scalar[T]] - Threshold for top-p sampling if is_top_p is True else min-p cofficient
        skip_sort: UnsafePointer[Scalar[DType.bool]] - Output buffer to store whether sorting is needed
    """
    tid = thread_idx.x
    bid = block_idx.x
    block_size = block_dim.x

    batch_id = bid // num_blocks_per_input
    block_lane = bid % num_blocks_per_input

    _in_buffer = in_buffer + batch_id * num_elements

    # # Allocate shared memory for the values and indices
    var topk_sram = external_memory[
        TopK_2[T, largest],
        address_space = AddressSpace.SHARED,
        alignment = alignof[TopK_2[T, largest]](),
    ]()

    # Pack the topk_vals and topk_idxs into shared memory
    var block_offset: UInt = block_lane * block_size
    var stride = block_size * num_blocks_per_input
    topk_sram[tid] = TopK_2[T, largest]()
    for i in range(tid + block_offset, num_elements, stride):
        topk_sram[tid].insert(_in_buffer[i], i)

    barrier()

    # Prepare for K iterations to find the local top-K elements
    for k in range(K):
        # Initialize each thread with its own TopK_2 value and index
        var partial = topk_sram[tid]

        # Perform block-level reduction to find the maximum TopK_2
        var total = _block_reduce_topk[T, largest](partial)

        if tid == 0:
            # Store the local top-K values and indices in global memory
            var vector_idx: UInt = total.p
            local_topk_vals[bid * K + k] = total.u
            local_topk_idxs[bid * K + k] = Scalar[DType.index](vector_idx).cast[
                out_idx_type
            ]()

            @parameter
            if is_top_p:
                # In top-p sampling, we check if the highest probability token exceeds
                # the probability threshold (p_threshold). If it does, we can skip sorting
                # since we'll just sample this token. Otherwise, we need to sort to find
                # all tokens that sum to p_threshold probability mass.
                skip_sort[batch_id] = (
                    total.u > p_threshold[batch_id]
                ) and not _test_sort
                # If we're testing sort, we can't skip sort
            else:
                # For min-p sampling, we calculate a dynamic threshold as:
                # threshold = min_p_coefficient * max_probability
                # This ensures we only consider tokens with probability at least
                # min_p_coefficient times the highest probability token.
                var p_threshold_val = p_threshold[batch_id] * total.u
                # update with actual min-p threshold
                p_threshold[batch_id] = p_threshold_val
                skip_sort[batch_id] = False

            # Remove the found maximum from consideration in the next iteration
            var orig_tid = (vector_idx - block_offset) % stride
            topk_sram[orig_tid].u = _topk_dead_val[T, largest]()

        barrier()


@always_inline
fn normalize(value: BFloat16) -> Scalar[DType.uint16]:
    @always_inline
    fn reinterpret(value: BFloat16) -> Scalar[DType.uint16]:
        # For unsigned integral types: No conversion needed, return as-is
        return bitcast[DType.uint16, 1](value)

    # Normalize bf16 values by flipping the sign bit for positive and fully
    # inverting negative numbers
    var bits = reinterpret(value)
    alias sign_bit_mask = (0b1 << (bitwidthof[DType.bfloat16]() - 1))
    if bits & sign_bit_mask:
        # For negative numbers, flip all bits (two's complement behavior)
        return ~bits
    else:
        # For positive numbers, flip only the sign bit
        return bits ^ sign_bit_mask


@always_inline
fn normalize_u32(value: UInt32) -> UInt32:
    return value


@always_inline
fn normalize(value: Int32) -> UInt32:
    @always_inline
    fn reinterpret(value: Int32) -> UInt32:
        # For signed integral types: Convert to unsigned int to ensure proper
        # comparison
        return value.cast[DType.uint32]()

    # For signed integers: Flip the most significant bit to ensure correct ordering
    # This makes negative numbers appear "smaller" than positive numbers in
    # unsigned comparison
    alias sign_bit_mask = (0b1 << (bitwidthof[DType.int32]() - 1))

    return reinterpret(value) ^ sign_bit_mask


@always_inline
fn normalize(value: Scalar[DType.uint16]) -> Scalar[DType.uint16]:
    return value


@always_inline
fn normalize(value: Float32) -> UInt32:
    @always_inline
    fn reinterpret(value: Float32) -> UInt32:
        # For floating-point types: Reinterpret the bit pattern as an unsigned int
        # This allows for comparison of floating-point values based on their binary
        # representation
        return bitcast[DType.uint32, 1](value)

    var bits = reinterpret(value)
    alias sign_bit = bitwidthof[DType.float32]() - 1
    # Flip all bits if the value is negative (sign bit is 1)
    # This makes more negative numbers appear "smaller" in unsigned comparison
    return bits ^ ((-(bits >> sign_bit)) | (0b1 << sign_bit))


@always_inline
fn normalize(
    value: Scalar,
    out result: Scalar[_uint_type_of_width[bitwidthof[value.dtype]()]()],
):
    """
    Normalize the value to the appropriate unsigned integer type. This is needed
    for radix sort to work correctly.
    """
    alias dtype = value.dtype

    @parameter
    if dtype is DType.int32:
        return normalize(rebind[Int32](value)).cast[result.dtype]()
    elif dtype is DType.uint32:
        return normalize(rebind[UInt32](value)).cast[result.dtype]()
    elif dtype is DType.float32:
        return normalize(rebind[Float32](value)).cast[result.dtype]()
    # TODO: These below don't return uint32 so must generalize and fix
    elif dtype is DType.uint16:
        return normalize(rebind[Scalar[DType.uint16]](value)).cast[
            result.dtype
        ]()
    elif dtype is DType.float16:
        return normalize(rebind[Float16](value)).cast[result.dtype]()
    elif dtype is DType.bfloat16:
        return normalize(rebind[BFloat16](value)).cast[result.dtype]()
    else:
        constrained[False, "unhandled normalize type"]()
        return 0


@always_inline
fn radix_sort_pairs_kernel[
    type: DType,
    out_idx_type: DType,
    current_bit: Int,
    ascending: Bool = False,
    BLOCK_SIZE: Int = 256,  # found empirically
    NUM_BITS_PER_PASS: Int = 4,
](
    input_keys_: UnsafePointer[Scalar[type]],  # modifies input
    output_keys_: UnsafePointer[Scalar[type]],
    input_key_ids_: UnsafePointer[Scalar[out_idx_type]],  # modifies input
    output_key_ids_: UnsafePointer[Scalar[out_idx_type]],
    num_keys: Int,
    skip_sort: UnsafePointer[Scalar[DType.bool]],
):
    """
    Radix pair sort kernel for (default) descending order.

    Parameters:
        type: DType - Data type.
        out_idx_type: DType - Output index type.
        current_bit: Int - Current bit to start sorting NUM_BITS_PER_PASS bits at.
        ascending: Bool - Whether to sort in ascending order.
        BLOCK_SIZE: Int - Block size.
        NUM_BITS_PER_PASS: Int - Number of bits per pass.

    Args:
        input_keys_: Input tensor values to sort.
        output_keys_: Output tensor values sorted in (default) descending order.
        input_key_ids_: Input tensor indices.
        output_key_ids_: Output tensor indices sorted in (default) descending order.
        num_keys: Number of keys to sort per batch.
        skip_sort: Whether sorting is skipped for this batch.

    Implementation based on:
    AMD. Introduction to GPU Radix Sort. GPUOpen, 2017. Available at:
    https://gpuopen.com/download/publications/Introduction_to_GPU_Radix_Sort.pdf.
    """

    var tid = thread_idx.x
    var batch_id = block_idx.x
    var elems_per_thread = ceildiv(num_keys, BLOCK_SIZE)
    alias NUM_BUCKETS = 2**NUM_BITS_PER_PASS

    var input_keys = input_keys_ + batch_id * num_keys
    var output_keys = output_keys_ + batch_id * num_keys
    var input_key_ids = input_key_ids_ + batch_id * num_keys
    var output_key_ids = output_key_ids_ + batch_id * num_keys

    if skip_sort[batch_id]:
        return

    # Shared mem declarations
    var s_counts = stack_allocation[
        BLOCK_SIZE * NUM_BUCKETS,
        Int32,
        address_space = AddressSpace.SHARED,
    ]()
    var total_counts = stack_allocation[
        NUM_BUCKETS,
        Int32,
        address_space = AddressSpace.SHARED,
    ]()
    var total_offsets = stack_allocation[
        (NUM_BUCKETS + 1),  # +1 extended size for descending
        Int32,
        address_space = AddressSpace.SHARED,
    ]()
    var total_offsets_descending = stack_allocation[
        NUM_BUCKETS,
        Int32,
        address_space = AddressSpace.SHARED,
    ]()
    var s_thread_offsets = stack_allocation[
        BLOCK_SIZE * NUM_BUCKETS,
        Int32,
        address_space = AddressSpace.SHARED,
    ]()

    # Initialize counts[NUM_BUCKETS]
    var counts_buf = NDBuffer[
        DType.int32, 1, MutableAnyOrigin, DimList(NUM_BUCKETS)
    ].stack_allocation()
    var counts = counts_buf.data
    counts_buf.fill(0)

    # Process elements and compute counts for each thread
    for index in range(tid * elems_per_thread, (tid + 1) * elems_per_thread):
        if index < num_keys:
            var key = input_keys[index]
            var normalized_key = normalize(key)
            var radix = (normalized_key >> current_bit) & (NUM_BUCKETS - 1)
            counts[radix] += 1

    # Store counts[NUM_BUCKETS] per thread into shared memory s_counts
    @parameter
    for i in range(NUM_BUCKETS):
        s_counts[tid * NUM_BUCKETS + i] = counts[i]
    barrier()

    # Compute total_counts[NUM_BUCKETS] by summing counts[NUM_BUCKETS] across threads
    if tid < NUM_BUCKETS:
        var sum = Int32(0)
        bucket_offset = tid

        @parameter
        for t in range(BLOCK_SIZE):
            sum += s_counts[t * NUM_BUCKETS + bucket_offset]
        total_counts[bucket_offset] = sum
    barrier()

    # Perform exclusive scan over total_counts[NUM_BUCKETS] to get total_offsets[NUM_BUCKETS]
    if tid == 0:
        total_offsets[0] = 0

        @parameter
        for i in range(1, NUM_BUCKETS + 1):
            total_offsets[i] = total_offsets[i - 1] + total_counts[i - 1]

    # Compute per-thread starting offsets per radix value
    @parameter
    for i in range(NUM_BUCKETS):
        s_thread_offsets[tid * NUM_BUCKETS + i] = s_counts[
            tid * NUM_BUCKETS + i
        ]
    barrier()

    # Perform exclusive scan over s_thread_offsets per radix value
    @parameter
    for radix in range(NUM_BUCKETS):
        # Initialize the offset to 1, which will be used to determine the distance
        # between threads whose values will be reduced/summed.
        var offset = 1
        while offset < BLOCK_SIZE:
            # Initialize a temporary variable to store the value from the neighboring thread.
            var val = Int32(0)
            if tid >= offset:
                # If the current thread ID is greater than or equal to the offset,
                # fetch the value from the neighboring thread that is 'offset' positions behind.
                val = s_thread_offsets[(tid - offset) * NUM_BUCKETS + radix]
            # Synchronize all threads to ensure that the value fetching is complete.
            barrier()
            # Add the fetched value to the current thread's value.
            s_thread_offsets[tid * NUM_BUCKETS + radix] += val
            # Synchronize all threads to ensure that the addition is complete.
            barrier()
            # Double the offset for the next iteration to fetch values from farther threads.
            offset <<= 1

        # After the loop, set the first thread's offset to 0.
        if tid == 0:
            s_thread_offsets[tid * NUM_BUCKETS + radix] = 0
        else:
            # For all other threads, set the offset to the value of the previous thread.
            s_thread_offsets[tid * NUM_BUCKETS + radix] = s_thread_offsets[
                (tid - 1) * NUM_BUCKETS + radix
            ]
        # Synchronize all threads to ensure that the final offset values are set.
        barrier()

    # Compute total_offsets_descending[NUM_BUCKETS] if needed
    @parameter
    if not ascending:
        if tid < NUM_BUCKETS:
            total_offsets_descending[tid] = (
                total_offsets[NUM_BUCKETS] - total_offsets[tid + 1]
            )
        barrier()

    # Each thread initializes local_offsets[NUM_BUCKETS] = 0
    var local_offsets_buf = NDBuffer[
        DType.int32, 1, MutableAnyOrigin, DimList(NUM_BUCKETS)
    ].stack_allocation()
    local_offsets_buf.fill(0)
    var local_offsets = local_offsets_buf.data

    # Now, each thread processes its elements, computes destination index, write to output
    for index in range(tid * elems_per_thread, (tid + 1) * elems_per_thread):
        if index < num_keys:
            var key = input_keys[index]
            var normalized_key = normalize(key)
            var radix = Int((normalized_key >> current_bit) & (NUM_BUCKETS - 1))

            # Adjust global_offset for ascending or descending order
            var global_offset: Int

            @parameter
            if ascending:
                global_offset = Int(
                    total_offsets[radix]
                    + s_thread_offsets[tid * NUM_BUCKETS + radix]
                    + local_offsets[radix]
                )
            else:
                global_offset = Int(
                    total_offsets_descending[radix]
                    + s_thread_offsets[tid * NUM_BUCKETS + radix]
                    + local_offsets[radix]
                )

            output_keys[global_offset] = key

            @parameter
            if current_bit == 0:
                output_key_ids[global_offset] = index
            else:
                output_key_ids[global_offset] = input_key_ids[index]

            local_offsets[radix] += 1


@always_inline
fn run_radix_sort_pairs_gpu[
    type: DType,
    out_idx_type: DType,
    rank: Int,
    ascending: Bool = False,
    BLOCK_SIZE: Int = 256,  # found empirically
    NUM_BITS_PER_PASS: Int = 4,
](
    ctx: DeviceContext,
    mut input_keys: NDBuffer[type, rank, MutableAnyOrigin],  # modifies input
    mut output_keys: NDBuffer[type, rank, MutableAnyOrigin],  # modifies output
    mut input_key_ids: NDBuffer[
        out_idx_type, rank, MutableAnyOrigin
    ],  # modifies input
    mut output_key_ids: NDBuffer[
        out_idx_type, rank, MutableAnyOrigin
    ],  # modifies output
    skip_sort: NDBuffer[DType.bool, rank],
) raises:
    var in_shape = input_keys.get_shape()
    var batch_size = in_shape[0]
    var vocab_size = in_shape[1]

    @parameter
    for current_bit in range(0, bitwidthof[type](), NUM_BITS_PER_PASS):
        alias kernel = radix_sort_pairs_kernel[
            type, out_idx_type, current_bit, ascending, BLOCK_SIZE
        ]

        ctx.enqueue_function[kernel](
            input_keys.data,
            output_keys.data,
            input_key_ids.data,
            output_key_ids.data,
            vocab_size,
            skip_sort.data,
            grid_dim=Dim(batch_size),
            block_dim=Dim(BLOCK_SIZE),
        )
        input_keys.data, output_keys.data = output_keys.data, input_keys.data

        var temp_key_ids = input_key_ids.data
        input_key_ids.data = output_key_ids.data
        output_key_ids.data = temp_key_ids

    output_keys.data = input_keys.data
    output_key_ids.data = input_key_ids.data


@always_inline
fn topp_minp_sampling_kernel[
    type: DType,
    out_idx_type: DType,
    is_top_p: Bool,
](
    p_thresholds_: UnsafePointer[Scalar[type]],
    sorted_probs_: UnsafePointer[Scalar[type]],
    sorted_ids_: UnsafePointer[Scalar[out_idx_type]],
    out_token_ids: UnsafePointer[Scalar[out_idx_type]],
    skip_sort: UnsafePointer[Scalar[DType.bool]],
    vocab_size: Int,
):
    """
    Top P-Min P sampling kernel.

    Parameters:
        type: DType - scalar values dtype.
        out_idx_type: DType - output index type.
        is_top_p: Bool - Whether to use Top-P (True) or Min-P (False) sampling.
    Args:
        p_thresholds_: Top p or min-p calculated thresholds for each batch.
        sorted_probs_: Sorted probabilities in descending order.
        sorted_ids_: Sorted token ids in descending order.
        out_token_ids: Output token ids.
        skip_sort: Whether sorting was skipped for this batch.
    """
    var tid = thread_idx.x
    var block_size = block_dim.x
    var batch_id = block_idx.x

    if skip_sort[batch_id]:
        # out_token_ids is already set by topk_wrapper
        return

    var p_threshold = p_thresholds_[batch_id]
    var sorted_probs = sorted_probs_ + batch_id * vocab_size
    var sorted_ids = sorted_ids_ + batch_id * vocab_size

    @parameter
    if is_top_p:
        if tid == 0:
            var rng_state = Random(seed=SEED)
            var rng = rng_state.step_uniform()
            var r = p_threshold * rng[0].cast[type]()
            for i in range(vocab_size):
                r -= sorted_probs[i]

                if r <= 0.0 or i == vocab_size - 1:

                    @parameter
                    if DEBUG_FILE:
                        print("sorted_probs[i]: ", sorted_probs[i])
                        print("r: ", r)
                        print("p_threshold: ", p_threshold)

                    out_token_ids[batch_id] = sorted_ids[i]
                    break
    else:
        # Min-P sampling
        if tid == 0:
            var rng_state = Random(seed=SEED)
            var rng = rng_state.step_uniform()

            # Step 1: Filter out tokens with probabilities less than the min-p threshold
            var sum_filtered_probs = Scalar[type](0.0)
            var num_filtered_tokens = 0
            for i in range(vocab_size):
                if sorted_probs[i] >= p_threshold:
                    sum_filtered_probs += sorted_probs[i]
                    num_filtered_tokens += 1
                else:
                    break

            # Step 2: Sample from normalized distribution of remaining tokens
            var r = sum_filtered_probs * rng[0].cast[type]()
            # Step 3: Select token based on normalized probabilities
            for i in range(num_filtered_tokens):
                r -= sorted_probs[i]

                if r <= 0.0 or i == vocab_size - 1:
                    out_token_ids[batch_id] = sorted_ids[i]

                    @parameter
                    if DEBUG_FILE:
                        print("sorted_probs[i]: ", sorted_probs[i])
                        print("r: ", r)
                        print("p_threshold: ", p_threshold)
                    break


@always_inline
fn _is_supported_type[type: DType]() -> Bool:
    """
    Check if the type is supported by the radix sort kernel.
    If not supported, need to add a normalize function for that
    numeric type.
    """
    if type in (DType.bfloat16, DType.float32):
        return True
    if type in (DType.uint16, DType.uint32, DType.int32):
        return True
    return False


@always_inline
fn _topp_minp_sampling_gpu[
    type: DType,
    rank: Int,
    out_idx_type: DType, //,
    is_top_p: Bool,
    _test_sort: Bool = False,
](
    ctx: DeviceContext,
    p_thresholds: NDBuffer[type, 1],
    input_logits: NDBuffer[type, rank],
    out_token_ids: NDBuffer[out_idx_type, rank],
    temperature: Scalar[type] = 1,
) raises:
    """
    GPU implementation of Top-P (nucleus) and Min-P sampling for token selection.
    This function applies temperature scaling, softmax, a radix sort, and then samples tokens
    based on either the cumulative probability mass (Top-P) or calculated probability threshold (Min-P).
    Token sampling algorithm details: https://www.notion.so/modularai/Token-sampler-1081044d37bb80c39932d6be9a4215d5


    Parameters:
        type: DType - The data type of the input logits, p_thresholds, and temperature.
        rank: Int - The rank of the input tensor (must be 2).
        out_idx_type: DType - The data type for output token indices.
        is_top_p: Bool - Whether to use Top-P (True) or Min-P (False) sampling. If Min-P, the
            p_thresholds are used as min-p coefficients that determine the minimum probability
            threshold for token inclusion.
        _test_sort: Bool - For internal testing purposes to check if the
            sorted probs are in descending order.
    Args:
        ctx: DeviceContext
            The context for GPU execution.
        p_thresholds: NDBuffer[type, 1]
            Batch of p values (thresholds) for Top-P/Min-P sampling.
            For Top-P: cumulative probability threshold (e.g., 0.9 means sample from top 90%).
            For Min-P: min-p coefficients that determine the minimum probability threshold.
        input_logits: NDBuffer[type, rank]
            Input logits tensor of shape [batch_size, vocab_size].
        out_token_ids: NDBuffer[out_idx_type, rank]
            Output buffer for sampled token indices of shape [batch_size, 1].
        temperature: Scalar[type]
            Temperature for softmax scaling of logits (default=1.0).
            Higher values increase diversity, lower values make sampling more deterministic.

    The implementation follows these steps:
    1. Apply temperature scaling to the input logits
    2. Convert logits to probabilities using softmax
    3. Sort probability/index pairs in descending order
    4. For each sequence in the batch:
        - For Top-P: Sample from tokens that sum to the p_threshold of probability mass
        - For Min-P: Sample from tokens that exceed the minimum probability threshold
    5. Output the selected token indices

    Based on sampling implementations from:
    - TensorRT-LLM: https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/kernels/samplingTopPKernels.cu#L199-L323
    - InternLM: https://github.com/InternLM/lmdeploy/
    """
    constrained[rank == 2, "Only rank 2 tensors are supported"]()
    constrained[_is_supported_type[type](), "Unsupported type"]()

    alias BLOCK_SIZE = 256

    # Step 1; Apply temperature scaling to the logits and apply
    # softmax to get probabilities
    var input_shape = input_logits.get_shape()
    var batch_size = input_shape[0]
    var vocab_size = input_shape[1]

    @parameter
    @__copy_capture(input_logits)
    fn apply_temperature[
        _simd_width: Int, _rank: Int
    ](coords: IndexList[_rank]) -> SIMD[type, _simd_width]:
        var val = input_logits.load[width=_simd_width](
            rebind[IndexList[rank]](coords)
        )
        return val / temperature

    # TODO: Should softmax be done in-place without needing this other buffer?
    var probs_buf = ctx.enqueue_create_buffer[type](
        input_shape.flattened_length()
    )
    var input_probs = NDBuffer[type, rank](probs_buf._unsafe_ptr(), input_shape)

    _softmax_gpu[
        type, 1, rank, DimList.create_unknown[rank](), apply_temperature
    ](input_shape, input_probs, rank - 1, ctx)

    # Step 2: Do a Top K=1 search on each vocab_size row of the
    #   probabilities tensor. This is to check if the most probable
    #   token exceeds P. If it does, we skip sorting by setting
    #   begin_offset_buf[bi] = offset_buf[bi]
    # materialize a vals buffer
    var max_vals_cache_buf = ctx.enqueue_create_buffer[type](Int(batch_size))
    var max_vals = NDBuffer[type, rank](
        max_vals_cache_buf._unsafe_ptr(), DimList(batch_size)
    )
    var skip_sort_buf = ctx.enqueue_create_buffer[DType.bool](Int(batch_size))
    var skip_sort = NDBuffer[DType.bool, rank](
        skip_sort_buf._unsafe_ptr(), DimList(batch_size)
    )

    alias K = 1
    alias num_blocks_per_input = 1
    ctx.enqueue_function[
        topk_wrapper[type, out_idx_type, is_top_p, _test_sort=_test_sort]
    ](
        K,
        vocab_size,
        num_blocks_per_input,
        input_probs.data,
        max_vals.data,
        out_token_ids.data,  # out_token_ids will now store the argmax
        p_thresholds.data,
        skip_sort.data,
        grid_dim=Dim(batch_size),
        block_dim=Dim(BLOCK_SIZE),
        shared_mem_bytes=_get_shmem_size_stg_1[type](BLOCK_SIZE),
    )

    # Step 3: Apply a global sort on the input tensor of probs
    # Create the input_ids buffer
    var sorted_probs_buf = ctx.enqueue_create_buffer[type](
        batch_size * vocab_size
    )
    var sorted_probs = NDBuffer[type, rank](
        sorted_probs_buf._unsafe_ptr(), DimList(batch_size, vocab_size)
    )
    var input_ids_buf = ctx.enqueue_create_buffer[out_idx_type](
        batch_size * vocab_size
    )
    var input_ids = NDBuffer[out_idx_type, rank](
        input_ids_buf._unsafe_ptr(), DimList(batch_size, vocab_size)
    )
    var sorted_ids_buf = ctx.enqueue_create_buffer[out_idx_type](
        batch_size * vocab_size
    )
    var sorted_ids = NDBuffer[out_idx_type, rank](
        sorted_ids_buf._unsafe_ptr(), DimList(batch_size, vocab_size)
    )

    run_radix_sort_pairs_gpu[BLOCK_SIZE=BLOCK_SIZE](
        ctx,
        input_probs,
        sorted_probs,
        input_ids,
        sorted_ids,
        skip_sort,
    )

    @parameter
    if _test_sort:
        # Copy output of sort & softmax back to original input tensor
        # for testing and debugging purposes
        ctx.enqueue_copy(
            input_logits.data, sorted_probs.data, input_shape.flattened_length()
        )

    # Step 4: Sample from the sorted probabilities by cumsumming
    ctx.enqueue_function[
        topp_minp_sampling_kernel[type, out_idx_type, is_top_p]
    ](
        p_thresholds.data,
        sorted_probs.data,
        sorted_ids.data,
        out_token_ids.data,
        skip_sort.data,
        vocab_size,
        grid_dim=Dim(batch_size),
        block_dim=Dim(BLOCK_SIZE),
    )
    _ = sorted_ids_buf^
    _ = sorted_probs_buf^
    _ = input_ids_buf^
    _ = max_vals_cache_buf^
    _ = skip_sort_buf^
    _ = probs_buf^


@always_inline
fn top_p_sampling_gpu[
    type: DType,
    rank: Int,
    out_idx_type: DType, //,
    _test_sort: Bool = False,
](
    ctx: DeviceContext,
    top_ps: NDBuffer[type, 1],
    input_logits: NDBuffer[type, rank],
    out_token_ids: NDBuffer[out_idx_type, rank],
    temperature: Scalar[type] = 1,
) raises:
    """
    GPU implementation of Top-P sampling for token selection.
    This function applies temperature scaling, softmax, a radix sort, and then
    samples tokens based on the cumulative probability mass (Top-P).
    """
    # TODO: Implement rank generalization
    constrained[rank == 2, "Only rank 2 tensors are supported"]()
    _topp_minp_sampling_gpu[is_top_p=True, _test_sort=_test_sort](
        ctx, top_ps, input_logits, out_token_ids, temperature
    )


@always_inline
fn min_p_sampling_gpu[
    type: DType,
    rank: Int,
    out_idx_type: DType, //,
    _test_sort: Bool = False,
](
    ctx: DeviceContext,
    min_ps: NDBuffer[type, 1],
    input_logits: NDBuffer[type, rank],
    out_token_ids: NDBuffer[out_idx_type, rank],
    temperature: Scalar[type] = 1,
) raises:
    """
    GPU implementation of Min-P sampling for token selection.
    This function applies temperature scaling, softmax, a radix sort, and then
    samples tokens based on the calculated probability threshold (Min-P).
    """
    _topp_minp_sampling_gpu[is_top_p=False, _test_sort=_test_sort](
        ctx, min_ps, input_logits, out_token_ids, temperature
    )
