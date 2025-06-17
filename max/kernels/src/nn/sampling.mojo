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

from algorithm.functional import elementwise
from layout import LayoutTensor
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

    @always_inline
    @parameter
    fn update_frequency_data_fn[width: Int, rank_: Int](idx: IndexList[rank_]):
        constrained[rank_ == 1, "update_frequency_data_fn: rank must be 1"]()

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
