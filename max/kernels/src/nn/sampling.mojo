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

from math import ceildiv, iota
from sys.info import simd_width_of

import gpu.primitives.block as block
from algorithm.functional import elementwise
from gpu import block_idx, thread_idx
from gpu.host.info import is_gpu
from layout._layout import TensorLayout
from layout._tile_tensor import TileTensor
from nn._ragged_utils import get_batch_from_row_offsets
from runtime.asyncrt import DeviceContextPtr

from utils import IndexList


fn apply_penalties_to_logits[
    logit_type: DType,
    penalty_type: DType,
    //,
    target: StaticString,
](
    logits: TileTensor[mut=True, logit_type, ...],
    compressed_frequency_data: TileTensor[DType.int32, ...],
    frequency_offsets: TileTensor[DType.uint32, ...],
    frequency_penalty: TileTensor[penalty_type, ...],
    presence_penalty: TileTensor[penalty_type, ...],
    repetition_penalty: TileTensor[penalty_type, ...],
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

    comptime assert frequency_offsets.flat_rank == 1
    comptime assert compressed_frequency_data.flat_rank == 2
    comptime assert repetition_penalty.flat_rank == 1
    comptime assert presence_penalty.flat_rank == 1
    comptime assert frequency_penalty.flat_rank == 1
    comptime assert logits.flat_rank == 2

    # all scalars
    comptime assert frequency_offsets.element_size == 1
    comptime assert compressed_frequency_data.element_size == 1
    comptime assert repetition_penalty.element_size == 1
    comptime assert presence_penalty.element_size == 1
    comptime assert frequency_penalty.element_size == 1
    comptime assert logits.element_size == 1

    @always_inline
    @parameter
    fn apply_penalties_fn[
        width: Int, rank_: Int, alignment: Int = 1
    ](idx: IndexList[rank_]):
        comptime assert rank_ == 1, "apply_penalties_fn: rank must be 1"

        var batch_id = get_batch_from_row_offsets(frequency_offsets, idx[0])
        var token = Int(compressed_frequency_data[idx[0], 0])

        var repetition_penalty_val = repetition_penalty[batch_id][0]
        var presence_penalty_val = presence_penalty[batch_id][0]
        var frequency_penalty_val = frequency_penalty[batch_id][0]
        # skip padding tokens
        if token >= 0:
            var count = compressed_frequency_data[idx[0], 1][0].cast[
                logit_type
            ]()

            var logit = logits[batch_id, token][0]

            if logit > 0:
                logit = logit / repetition_penalty_val.cast[logit_type]()
            else:
                logit = logit * repetition_penalty_val.cast[logit_type]()

            logit -= (
                frequency_penalty_val[0].cast[logit_type]() * count
                + presence_penalty_val[0].cast[logit_type]()
            )

            logits[batch_id, token] = logit

    var dispatch_shape = IndexList[1](Int(compressed_frequency_data.dim[0]()))
    elementwise[
        func=apply_penalties_fn,
        simd_width=1,
        target=target,
        _trace_description="apply_penalties_to_logits",
    ](dispatch_shape, ctx)


fn update_frequency_data_kernel[
    freq_data_origin: MutOrigin,
    FreqDataLayoutType: TensorLayout,
    freq_offsets_origin: ImmutOrigin,
    FreqOffsetsLayoutType: TensorLayout,
    new_tokens_origin: ImmutOrigin,
    NewTokensLayoutType: TensorLayout,
    token_type: DType,
    block_size: Int,
](
    compressed_frequency_data: TileTensor[
        DType.int32, FreqDataLayoutType, freq_data_origin
    ],
    frequency_offsets: TileTensor[
        DType.uint32, FreqOffsetsLayoutType, freq_offsets_origin
    ],
    new_tokens: TileTensor[token_type, NewTokensLayoutType, new_tokens_origin],
):
    """
    GPU kernel to update token frequency data in CSR format.

    Searches for new tokens in existing frequency data and either increments
    their count or adds them to the first available padding slot.
    """

    comptime assert frequency_offsets.flat_rank == 1
    comptime assert compressed_frequency_data.flat_rank == 2
    comptime assert new_tokens.flat_rank == 1

    comptime simd_width = simd_width_of[DType.int32]()
    comptime PADDING_TOKEN = -1

    var tid = thread_idx.x
    var batch_id = block_idx.x

    var tok_start = Int(frequency_offsets[batch_id])
    var tok_end = Int(frequency_offsets[batch_id + 1])
    var new_token = new_tokens[batch_id].cast[DType.int32]()

    var num_scans = ceildiv(tok_end - tok_start, block_size * simd_width)

    # search if the new token is already in the frequency data
    for scan_idx in range(num_scans):
        var tok_idx = tok_start + Int(
            (tid + UInt(scan_idx * block_size)) * UInt(simd_width)
        )

        var val = SIMD[DType.int32, simd_width](0)

        @parameter
        for i in range(simd_width):
            if tok_idx + i < tok_end:
                val[i] = compressed_frequency_data[tok_idx + i, 0]
            else:
                val[i] = Int32.MAX_FINITE

        var if_found = val.eq(new_token).select(
            iota[DType.int32, simd_width](Int32(tok_idx)),
            SIMD[DType.int32, simd_width](Int32.MIN_FINITE),
        )
        var first_padding_idx = val.eq(PADDING_TOKEN).select(
            iota[DType.int32, simd_width](Int32(tok_idx)),
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
    token_type: DType,
    //,
    target: StaticString,
](
    compressed_frequency_data: TileTensor[
        mut=True, DType.int32, address_space = AddressSpace.GENERIC, ...
    ],
    frequency_offsets: TileTensor[
        DType.uint32, address_space = AddressSpace.GENERIC, ...
    ],
    new_tokens: TileTensor[
        token_type, address_space = AddressSpace.GENERIC, ...
    ],
    ctx: DeviceContextPtr,
) raises:
    """
    Update the frequency data for the given new tokens.

    The frequency data is stored in a CSR format. This kernel expects there will be
    enough padding for each sequence to store the new tokens.
    """
    comptime assert frequency_offsets.flat_rank == 1
    comptime assert compressed_frequency_data.flat_rank == 2
    comptime assert new_tokens.flat_rank == 1
    comptime assert compressed_frequency_data.element_size == 1
    comptime assert new_tokens.element_size == 1

    @parameter
    if is_gpu[target]():
        comptime block_size = 128

        dev_ctx = ctx.get_device_context()
        comptime kernel = update_frequency_data_kernel[
            freq_data_origin = compressed_frequency_data.origin,
            FreqDataLayoutType = compressed_frequency_data.LayoutType,
            freq_offsets_origin = ImmutOrigin(frequency_offsets.origin),
            FreqOffsetsLayoutType = frequency_offsets.LayoutType,
            new_tokens_origin = ImmutOrigin(new_tokens.origin),
            NewTokensLayoutType = new_tokens.LayoutType,
            token_type=token_type,
            block_size=block_size,
        ]
        dev_ctx.enqueue_function_experimental[kernel](
            compressed_frequency_data,
            frequency_offsets.as_immut(),
            new_tokens.as_immut(),
            grid_dim=new_tokens.dim[0](),
            block_dim=block_size,
        )

    else:

        @always_inline
        @parameter
        fn update_frequency_data_fn[
            width: Int, rank_: Int, alignment: Int = 1
        ](idx: IndexList[rank_]):
            comptime assert (
                rank_ == 1
            ), "update_frequency_data_fn: rank must be 1"

            var tok_start = frequency_offsets[idx[0]]
            var tok_end = frequency_offsets[idx[0] + 1]

            var new_token = new_tokens[idx[0]][0].cast[DType.int32]()

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

        var dispatch_shape = IndexList[1](new_tokens.numel())
        elementwise[
            func=update_frequency_data_fn,
            simd_width=1,
            target=target,
            _trace_description="update_frequency_data",
        ](dispatch_shape, ctx)
