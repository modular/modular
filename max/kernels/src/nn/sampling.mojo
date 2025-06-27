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

from math import ceildiv, iota
from sys.info import simdwidthof

import gpu.block as block
from algorithm.functional import elementwise
from gpu import block_idx, thread_idx, WARP_SIZE
from gpu.host.info import is_gpu
from layout import Layout, LayoutTensor
from nn._ragged_utils import get_batch_from_row_offsets
from runtime.asyncrt import DeviceContextPtr

from utils import IndexList


fn apply_penalties_to_logits[
    logit_type: DType,
    penalty_type: DType, //,
    target: StaticString,
](
    logits: LayoutTensor[mut=True, logit_type, **_],
    compressed_frequency_data: LayoutTensor[DType.int32, **_],
    frequency_offsets: LayoutTensor[DType.uint32, **_],
    frequency_penalty: LayoutTensor[penalty_type, **_],
    presence_penalty: LayoutTensor[penalty_type, **_],
    repetition_penalty: LayoutTensor[penalty_type, **_],
    ctx: DeviceContextPtr,
) raises:
    """
    Apply penalties to the logits based on the frequency of the tokens in the batch.

    The frequency data is stored in a CSR format, where the frequency_offsets is the
    starting index of each sequence in the frequency_data array. The frequency_data
    array is a 2D array, where:
    - frequency_data[i, 0] is the token id
    - frequency_data[i, 1] is the frequency of the token in the sequence
    """

    @always_inline
    @parameter
    fn apply_penalties_fn[width: Int, rank_: Int](idx: IndexList[rank_]):
        constrained[rank_ == 1, "apply_penalties_fn: rank must be 1"]()

        var batch_id = get_batch_from_row_offsets(frequency_offsets, idx[0])
        var token = Int(compressed_frequency_data[idx[0], 0])

        var repetition_penalty_val = repetition_penalty[batch_id][0]
        var presence_penalty_val = presence_penalty[batch_id][0]
        var frequency_penalty_val = frequency_penalty[batch_id][0]
        # skip padding tokens
        if token >= 0:
            var count = rebind[Scalar[DType.int32]](
                compressed_frequency_data[idx[0], 1]
            ).cast[logit_type]()

            var logit = logits[batch_id, token]

            if logit > 0:
                logit = logit / repetition_penalty_val.cast[logit_type]()
            else:
                logit = logit * repetition_penalty_val.cast[logit_type]()

            logit -= (
                frequency_penalty_val.cast[logit_type]() * count
                + presence_penalty_val.cast[logit_type]()
            )

            logits[batch_id, token] = logit

    var dispatch_shape = IndexList[1](compressed_frequency_data.dim[0]())
    elementwise[
        func=apply_penalties_fn,
        simd_width=1,
        target=target,
        _trace_description="apply_penalties_to_logits",
    ](dispatch_shape, ctx)


fn update_frequency_data_kernel[
    token_type: DType,
    block_size: Int,
    freq_data_layout: Layout,
    freq_offsets_layout: Layout,
    new_tokens_layout: Layout,
](
    compressed_frequency_data: LayoutTensor[
        mut=True, DType.int32, freq_data_layout, MutableAnyOrigin
    ],
    frequency_offsets: LayoutTensor[
        DType.uint32, freq_offsets_layout, MutableAnyOrigin
    ],
    new_tokens: LayoutTensor[token_type, new_tokens_layout, MutableAnyOrigin],
):
    """
    GPU kernel to update token frequency data in CSR format.

    Searches for new tokens in existing frequency data and either increments
    their count or adds them to the first available padding slot.
    """

    alias simd_width = simdwidthof[DType.int32]()
    alias PADDING_TOKEN = -1

    var tid = thread_idx.x
    var batch_id = block_idx.x

    var tok_start = Int(frequency_offsets[batch_id])
    var tok_end = Int(frequency_offsets[batch_id + 1])
    var new_token = rebind[Scalar[token_type]](new_tokens[batch_id]).cast[
        DType.int32
    ]()

    var num_scans = ceildiv(tok_end - tok_start, block_size * simd_width)

    # search if the new token is already in the frequency data
    for scan_idx in range(num_scans):
        var tok_idx = tok_start + (tid + scan_idx * block_size) * simd_width

        var val = SIMD[DType.int32, simd_width](0)

        @parameter
        for i in range(simd_width):
            if tok_idx + i < tok_end:
                val[i] = rebind[Int32](
                    compressed_frequency_data[tok_idx + i, 0]
                )
            else:
                val[i] = Int32.MAX_FINITE

        var if_found = (val == new_token).select(
            iota[DType.int32, simd_width](tok_idx),
            SIMD[DType.int32, simd_width](Int32.MIN_FINITE),
        )
        var first_padding_idx = (val == PADDING_TOKEN).select(
            iota[DType.int32, simd_width](tok_idx),
            SIMD[DType.int32, simd_width](Int32.MAX_FINITE),
        )

        var target_token_idx = block.max[block_size=block_size, broadcast=True](
            if_found.reduce_max()
        )
        var padding_token_idx = block.min[
            block_size=block_size, broadcast=True
        ](first_padding_idx.reduce_min())

        if target_token_idx != Int32.MIN_FINITE:
            # we found the target token, update the frequency data
            if tid == 0:
                compressed_frequency_data[Int(target_token_idx), 1] += 1
            return
        elif padding_token_idx != Int32.MAX_FINITE:
            # we don't find the target token, but we found a padding token
            if tid == 0:
                compressed_frequency_data[Int(padding_token_idx), 0] = new_token
                compressed_frequency_data[Int(padding_token_idx), 1] = 1
            return


fn update_frequency_data[
    token_type: DType, //,
    target: StaticString,
](
    compressed_frequency_data: LayoutTensor[mut=True, DType.int32, **_],
    frequency_offsets: LayoutTensor[DType.uint32, **_],
    new_tokens: LayoutTensor[token_type, **_],
    ctx: DeviceContextPtr,
) raises:
    """
    Update the frequency data for the given new tokens.

    The frequency data is stored in a CSR format. This kernel expects there will be
    enough padding for each sequence to store the new tokens.
    """

    @parameter
    if is_gpu[target]():
        alias block_size = 128

        dev_ctx = ctx.get_device_context()
        dev_ctx.enqueue_function[
            update_frequency_data_kernel[
                token_type,
                block_size,
                compressed_frequency_data.layout,
                frequency_offsets.layout,
                new_tokens.layout,
            ]
        ](
            compressed_frequency_data,
            frequency_offsets,
            new_tokens,
            grid_dim=new_tokens.dim[0](),
            block_dim=block_size,
        )

    else:

        @always_inline
        @parameter
        fn update_frequency_data_fn[
            width: Int, rank_: Int
        ](idx: IndexList[rank_]):
            constrained[
                rank_ == 1, "update_frequency_data_fn: rank must be 1"
            ]()

            var tok_start = frequency_offsets[idx[0]]
            var tok_end = frequency_offsets[idx[0] + 1]

            var new_token = rebind[Scalar[token_type]](new_tokens[idx[0]]).cast[
                DType.int32
            ]()

            for tok_id in range(tok_start, tok_end):
                if compressed_frequency_data[tok_id, 0] == new_token:
                    compressed_frequency_data[tok_id, 1] += 1
                    break

                # if we encounter a padding token, add the new token to the
                # occurrences tensor
                elif compressed_frequency_data[tok_id, 0] == -1:
                    compressed_frequency_data[tok_id, 0] = new_token
                    compressed_frequency_data[tok_id, 1] = 1
                    break

        var dispatch_shape = IndexList[1](new_tokens.size())
        elementwise[
            func=update_frequency_data_fn,
            simd_width=1,
            target=target,
            _trace_description="update_frequency_data",
        ](dispatch_shape, ctx)
