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


from math import iota
from random import random_float64

from buffer import NDBuffer
from buffer.dimlist import DimList
from memory import UnsafePointer
from nn.softmax import softmax

from utils import IndexList


@always_inline
fn top_p_sampling[
    type: DType,
    rank: Int,
    out_idx_type: DType, //,
    _test_sort: Bool = False,
](
    top_ps: NDBuffer[type, 1],
    input_logits: NDBuffer[mut=True, type, rank],
    out_token_ids: NDBuffer[mut=True, out_idx_type, rank],
    temperature: Scalar[type] = 1,
) raises:
    """
    Naive CPU implementation of Top-P sampling for token selection.
    This function applies temperature scaling, softmax, a merge sort, and then
    samples tokens based on the cumulative probability mass (Top-P).
    """
    # TODO: Implement rank generalization
    constrained[rank == 2, "Only rank 2 tensors are supported"]()
    _topp_minp_sampling[is_top_p=True, _test_sort=_test_sort](
        top_ps, input_logits, out_token_ids, temperature
    )


@always_inline
fn min_p_sampling[
    type: DType,
    rank: Int,
    out_idx_type: DType, //,
    _test_sort: Bool = False,
](
    min_ps: NDBuffer[type, 1],
    input_logits: NDBuffer[mut=True, type, rank],
    out_token_ids: NDBuffer[mut=True, out_idx_type, rank],
    temperature: Scalar[type] = 1,
) raises:
    """
    Naive CPU implementation of Min-P sampling for token selection.
    This function applies temperature scaling, softmax, a merge sort, and then
    samples tokens based on the calculated probability threshold (Min-P).
    """
    _topp_minp_sampling[is_top_p=False, _test_sort=_test_sort](
        min_ps, input_logits, out_token_ids, temperature
    )


@always_inline
fn _topp_minp_sampling[
    type: DType,
    rank: Int,
    out_idx_type: DType, //,
    is_top_p: Bool,
    _test_sort: Bool = False,
](
    p_thresholds: NDBuffer[type, 1],
    input_logits: NDBuffer[mut=True, type, rank],
    out_token_ids: NDBuffer[mut=True, out_idx_type, rank],
    temperature: Scalar[type] = 1,
) raises:
    """
    Naive CPU implementation of Top-P/Min-P sampling for token selection.
    This function applies temperature scaling, softmax, a merge sort, and then
    samples tokens based on either cumulative probability mass (Top-P) or
    minimum probability threshold (Min-P).

    Parameters:
        type: DType - The data type of the input logits, p_thresholds, and temperature.
        rank: Int - The rank of the input tensor (must be 2).
        out_idx_type: DType - The data type for output token indices.
        is_top_p: Bool - Whether to use Top-P (True) or Min-P (False) sampling.
        _test_sort: Bool - For internal testing purposes to check if the
            sorted probs are in descending order. If true, copies the sorted
            probs back into input_logits.
    Args:
        p_thresholds: NDBuffer[type, 1] - Sampling thresholds, one per batch.
        input_logits: NDBuffer[type, rank] - Input logits (modified in-place).
        out_token_ids: NDBuffer[out_idx_type, rank] - Output sampled token IDs.
        temperature: Scalar[type] - Temperature for logits scaling.
    """
    constrained[rank == 2, "Only rank 2 tensors are supported"]()
    var input_shape = input_logits.get_shape()
    var batch_size = input_shape[0]
    var vocab_size = input_shape[1]

    var sorted_probs_ptr = UnsafePointer[Scalar[type]].alloc(
        batch_size * vocab_size
    )
    var sorted_probs = NDBuffer[type, rank](
        sorted_probs_ptr, DimList(batch_size, vocab_size)
    )

    var sorted_ids_ptr = UnsafePointer[Scalar[out_idx_type]].alloc(
        batch_size * vocab_size
    )
    var sorted_ids = NDBuffer[out_idx_type, rank](
        sorted_ids_ptr, DimList(batch_size, vocab_size)
    )

    # Initialize sorted_ids with iota values
    for batch_id in range(batch_size):
        iota(sorted_ids.data + (batch_id * vocab_size), vocab_size)
        # Copy input_logits to sorted_probs
        for i in range(vocab_size):
            var batch_offset = batch_id * vocab_size
            sorted_probs.data[batch_offset + i] = input_logits.data[
                batch_offset + i
            ]

    @parameter
    @__copy_capture(input_logits)
    fn apply_temperature[
        _simd_width: Int, _rank: Int
    ](coords: IndexList[_rank]) -> SIMD[type, _simd_width]:
        var val = input_logits.load[width=_simd_width](
            rebind[IndexList[rank]](coords)
        )
        return val / temperature

    softmax[simd_width=1, input_fn=apply_temperature](
        input_logits.get_shape(),
        sorted_probs,
        axis=rank - 1,
    )

    sort_buf_descending(sorted_probs, sorted_ids, vocab_size)

    # Copy sorted probs back to input_logits if testing
    @parameter
    if _test_sort:
        for i in range(batch_size * vocab_size):
            input_logits.data[i] = sorted_probs.data[i]

    # Process each batch
    for batch in range(batch_size):
        var p_threshold = p_thresholds[batch]

        @parameter
        if is_top_p:
            # Sample using top-p (nucleus) sampling
            var r = p_threshold * random_float64().cast[type]()
            for i in range(vocab_size):
                r -= sorted_probs[batch, i]
                if r <= 0 or i == vocab_size - 1:
                    out_token_ids[batch, 0] = sorted_ids[batch, i]
                    break
        else:
            # Sample using min-p sampling
            # Step 1: Filter out tokens with probabilities less than min-p threshold
            var sum_filtered_probs = Scalar[type](0.0)
            var num_filtered_tokens = 0
            for i in range(vocab_size):
                if sorted_probs[batch, i] >= p_threshold:
                    sum_filtered_probs += sorted_probs[batch, i]
                    num_filtered_tokens += 1
                else:
                    break

            # Step 2: Sample from normalized distribution of remaining tokens
            var r = sum_filtered_probs * random_float64().cast[type]()

            # Step 3: Select token based on normalized probabilities
            for i in range(num_filtered_tokens):
                r -= sorted_probs[batch, i]
                if r <= 0 or i == vocab_size - 1:
                    out_token_ids[batch, 0] = sorted_ids[batch, i]
                    break

    sorted_ids_ptr.free()
    sorted_probs_ptr.free()


@always_inline
fn sort_buf_descending[
    type: DType,
    out_idx_type: DType,
    rank: Int, //,
](
    mut buf_keys: NDBuffer[mut=True, type, rank],
    mut buf_ids: NDBuffer[mut=True, out_idx_type, rank],
    vocab_size: Int,
):
    """Sort each batch separately in descending order using parallel merge sort.
    """
    constrained[rank == 2, "rank must be 2"]()
    var batch_size = buf_keys.num_elements() // vocab_size

    for batch_id in range(batch_size):
        var start = batch_id * vocab_size
        var end = start + vocab_size
        merge_sort_recursive(buf_keys, buf_ids, start, end)


fn merge_sort_recursive[
    type: DType, out_idx_type: DType, rank: Int
](
    mut buf_keys: NDBuffer[mut=True, type, rank],
    mut buf_ids: NDBuffer[mut=True, out_idx_type, rank],
    start: Int,
    end: Int,
):
    """Recursive merge sort implementation."""
    if end - start > 1:
        var mid = start + (end - start) // 2
        merge_sort_recursive(buf_keys, buf_ids, start, mid)
        merge_sort_recursive(buf_keys, buf_ids, mid, end)
        merge(buf_keys, buf_ids, start, mid, end)


@always_inline
fn merge[
    type: DType, out_idx_type: DType, rank: Int
](
    mut buf_keys: NDBuffer[mut=True, type, rank],
    mut buf_ids: NDBuffer[mut=True, out_idx_type, rank],
    start: Int,
    mid: Int,
    end: Int,
):
    """Merge two sorted subarrays into one sorted array."""
    var left_size = mid - start
    var right_size = end - mid

    # Create temporary arrays
    var left_keys_ptr = UnsafePointer[Scalar[type]].alloc(left_size)
    var right_keys_ptr = UnsafePointer[Scalar[type]].alloc(right_size)
    var left_ids_ptr = UnsafePointer[Scalar[out_idx_type]].alloc(left_size)
    var right_ids_ptr = UnsafePointer[Scalar[out_idx_type]].alloc(right_size)

    # Copy data to temporary arrays
    for i in range(left_size):
        left_keys_ptr[i] = buf_keys.data[start + i]
        left_ids_ptr[i] = buf_ids.data[start + i]
    for i in range(right_size):
        right_keys_ptr[i] = buf_keys.data[mid + i]
        right_ids_ptr[i] = buf_ids.data[mid + i]

    # Merge back into original array
    var i = 0  # Index for left subarray
    var j = 0  # Index for right subarray
    var k = start  # Index for merged array

    while i < left_size and j < right_size:
        if left_keys_ptr[i] >= right_keys_ptr[j]:  # Use >= for descending order
            buf_keys.data[k] = left_keys_ptr[i]
            buf_ids.data[k] = left_ids_ptr[i]
            i += 1
        else:
            buf_keys.data[k] = right_keys_ptr[j]
            buf_ids.data[k] = right_ids_ptr[j]
            j += 1
        k += 1

    # Copy remaining elements if any
    while i < left_size:
        buf_keys.data[k] = left_keys_ptr[i]
        buf_ids.data[k] = left_ids_ptr[i]
        i += 1
        k += 1

    while j < right_size:
        buf_keys.data[k] = right_keys_ptr[j]
        buf_ids.data[k] = right_ids_ptr[j]
        j += 1
        k += 1

    # Free temporary arrays
    left_keys_ptr.free()
    right_keys_ptr.free()
    left_ids_ptr.free()
    right_ids_ptr.free()
