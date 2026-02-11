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


from math import iota

from random import random_float64
from layout._coord import Coord, Idx, coord_to_index_list
from layout._layout import row_major
from layout._tile_tensor import TileTensor
from nn.softmax import softmax

from utils import IndexList


@always_inline
fn top_p_sampling[
    dtype: DType,
    out_idx_type: DType,
    //,
    _test_sort: Bool = False,
](
    top_ps: TileTensor[dtype, ...],
    input_logits: TileTensor[mut=True, dtype, ...],
    out_token_ids: TileTensor[mut=True, out_idx_type, ...],
    temperature: Scalar[dtype] = 1,
) raises:
    """
    Naive CPU implementation of Top-P sampling for token selection.
    This function applies temperature scaling, softmax, a merge sort, and then
    samples tokens based on the cumulative probability mass (Top-P).
    """
    # TODO: Implement rank generalization
    comptime assert input_logits.rank == 2, "Only rank 2 tensors are supported"
    _topp_minp_sampling[is_top_p=True, _test_sort=_test_sort](
        top_ps, input_logits, out_token_ids, temperature
    )


@always_inline
fn min_p_sampling[
    dtype: DType,
    out_idx_type: DType,
    //,
    _test_sort: Bool = False,
](
    min_ps: TileTensor[dtype, ...],
    input_logits: TileTensor[mut=True, dtype, ...],
    out_token_ids: TileTensor[mut=True, out_idx_type, ...],
    temperature: Scalar[dtype] = 1,
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
    dtype: DType,
    out_idx_type: DType,
    //,
    is_top_p: Bool,
    _test_sort: Bool = False,
](
    p_thresholds: TileTensor[dtype, ...],
    input_logits: TileTensor[mut=True, dtype, ...],
    out_token_ids: TileTensor[mut=True, out_idx_type, ...],
    temperature: Scalar[dtype] = 1,
) raises:
    """
    Naive CPU implementation of Top-P/Min-P sampling for token selection.
    This function applies temperature scaling, softmax, a merge sort, and then
    samples tokens based on either cumulative probability mass (Top-P) or
    minimum probability threshold (Min-P).

    Parameters:
        dtype: DType - The data type of the input logits, p_thresholds, and temperature.
        out_idx_type: DType - The data type for output token indices.
        is_top_p: Bool - Whether to use Top-P (True) or Min-P (False) sampling.
        _test_sort: Bool - For internal testing purposes to check if the
            sorted probs are in descending order. If true, copies the sorted
            probs back into input_logits.
    Args:
        p_thresholds: NDBuffer[dtype, 1] - Sampling thresholds, one per batch.
        input_logits: NDBuffer[dtype, rank] - Input logits (modified in-place).
        out_token_ids: NDBuffer[out_idx_type, rank] - Output sampled token IDs.
        temperature: Scalar[dtype] - Temperature for logits scaling.
    """
    comptime assert (
        input_logits.flat_rank == 2
    ), "Only rank 2 tensors are supported"
    comptime assert (
        out_token_ids.flat_rank == 2
    ), "Only rank 2 tensors are supported"
    comptime assert p_thresholds.flat_rank == 1
    var input_shape = coord_to_index_list(input_logits.layout.shape_coord())
    var batch_size = input_shape[0]
    var vocab_size = input_shape[1]

    var sorted_probs_ptr = alloc[Scalar[dtype]](batch_size * vocab_size)
    var sorted_probs = TileTensor(
        sorted_probs_ptr,
        row_major(Coord(Idx(batch_size), Idx(vocab_size))),
    )

    var sorted_ids_ptr = alloc[Scalar[out_idx_type]](batch_size * vocab_size)
    var sorted_ids = TileTensor(
        sorted_ids_ptr,
        row_major(Coord(Idx(batch_size), Idx(vocab_size))),
    )

    comptime assert sorted_probs.element_size == out_token_ids.element_size
    comptime assert out_token_ids.element_size == 1

    # Initialize sorted_ids with iota values
    for batch_id in range(batch_size):
        iota(sorted_ids.ptr + (batch_id * vocab_size), vocab_size)
        # Copy input_logits to sorted_probs
        for i in range(vocab_size):
            var batch_offset = batch_id * vocab_size
            sorted_probs.ptr[batch_offset + i] = input_logits.ptr[
                batch_offset + i
            ]

    @parameter
    @__copy_capture(input_logits)
    fn apply_temperature[
        _simd_width: Int, _rank: Int
    ](coords: IndexList[_rank]) -> SIMD[dtype, _simd_width]:
        var i = input_logits.layout(Coord(coords))
        var val = input_logits.ptr.load[width=_simd_width](i)
        return val / temperature

    var shape = IndexList[input_logits.rank]()

    @parameter
    for i in range(input_logits.rank):
        shape[i] = input_logits.layout.shape[i]().value()

    softmax[simd_width=1, input_fn=apply_temperature](
        shape,
        sorted_probs.to_layout_tensor(),
        axis=input_logits.rank - 1,
    )

    sort_buf_descending(sorted_probs, sorted_ids, vocab_size)

    # Copy sorted probs back to input_logits if testing
    @parameter
    if _test_sort:
        for i in range(batch_size * vocab_size):
            input_logits.ptr[i] = sorted_probs.ptr[i]

    # Process each batch
    for batch in range(batch_size):
        var p_threshold = p_thresholds[batch]

        @parameter
        if is_top_p:
            # Sample using top-p (nucleus) sampling
            var r = p_threshold * random_float64().cast[dtype]()
            for i in range(vocab_size):
                r -= sorted_probs[batch, i]
                if r <= 0 or i == vocab_size - 1:
                    sid = sorted_ids[batch, i]
                    out_token_ids[batch, 0] = sid
                    break
        else:
            # Sample using min-p sampling
            # Step 1: Filter out tokens with probabilities less than min-p threshold
            var sum_filtered_probs = SIMD[dtype, out_token_ids.element_size](
                0.0
            )
            var num_filtered_tokens = 0
            for i in range(vocab_size):
                if sorted_probs[batch, i][0] >= p_threshold[0]:
                    sum_filtered_probs += sorted_probs[batch, i]
                    num_filtered_tokens += 1
                else:
                    break

            # Step 2: Sample from normalized distribution of remaining tokens
            var r = sum_filtered_probs * random_float64().cast[dtype]()

            # Step 3: Select token based on normalized probabilities
            for i in range(num_filtered_tokens):
                r -= sorted_probs[batch, i]
                if r <= 0 or i == vocab_size - 1:
                    sid = sorted_ids[batch, i]
                    out_token_ids[batch, 0] = sid
                    break

    sorted_ids_ptr.free()
    sorted_probs_ptr.free()


@always_inline
fn sort_buf_descending[
    dtype: DType, out_idx_type: DType
](
    mut buf_keys: TileTensor[mut=True, dtype, ...],
    mut buf_ids: TileTensor[mut=True, out_idx_type, ...],
    vocab_size: Int,
):
    """Sort each batch separately in descending order using parallel merge sort.
    """
    comptime assert buf_keys.rank == 2, "rank must be 2"
    var batch_size = buf_keys.numel() // vocab_size

    for batch_id in range(batch_size):
        var start = batch_id * vocab_size
        var end = start + vocab_size
        merge_sort_recursive(buf_keys, buf_ids, start, end)


fn merge_sort_recursive[
    dtype: DType,
    out_idx_type: DType,
](
    mut buf_keys: TileTensor[mut=True, dtype, ...],
    mut buf_ids: TileTensor[mut=True, out_idx_type, ...],
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
    dtype: DType, out_idx_type: DType
](
    mut buf_keys: TileTensor[mut=True, dtype, ...],
    mut buf_ids: TileTensor[mut=True, out_idx_type, ...],
    start: Int,
    mid: Int,
    end: Int,
):
    """Merge two sorted subarrays into one sorted array."""
    var left_size = mid - start
    var right_size = end - mid

    # Create temporary arrays
    var left_keys_ptr = alloc[Scalar[dtype]](left_size)
    var right_keys_ptr = alloc[Scalar[dtype]](right_size)
    var left_ids_ptr = alloc[Scalar[out_idx_type]](left_size)
    var right_ids_ptr = alloc[Scalar[out_idx_type]](right_size)

    # Copy data to temporary arrays
    for i in range(left_size):
        left_keys_ptr[i] = buf_keys.ptr[start + i]
        left_ids_ptr[i] = buf_ids.ptr[start + i]
    for i in range(right_size):
        right_keys_ptr[i] = buf_keys.ptr[mid + i]
        right_ids_ptr[i] = buf_ids.ptr[mid + i]

    # Merge back into original array
    var i = 0  # Index for left subarray
    var j = 0  # Index for right subarray
    var k = start  # Index for merged array

    while i < left_size and j < right_size:
        if left_keys_ptr[i] >= right_keys_ptr[j]:  # Use >= for descending order
            buf_keys.ptr[k] = left_keys_ptr[i]
            buf_ids.ptr[k] = left_ids_ptr[i]
            i += 1
        else:
            buf_keys.ptr[k] = right_keys_ptr[j]
            buf_ids.ptr[k] = right_ids_ptr[j]
            j += 1
        k += 1

    # Copy remaining elements if any
    while i < left_size:
        buf_keys.ptr[k] = left_keys_ptr[i]
        buf_ids.ptr[k] = left_ids_ptr[i]
        i += 1
        k += 1

    while j < right_size:
        buf_keys.ptr[k] = right_keys_ptr[j]
        buf_ids.ptr[k] = right_ids_ptr[j]
        j += 1
        k += 1

    # Free temporary arrays
    left_keys_ptr.free()
    right_keys_ptr.free()
    left_ids_ptr.free()
    right_ids_ptr.free()
