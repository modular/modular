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
from layout import LayoutTensor, Layout
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
    frequency_penalty: Scalar[penalty_type],
    presence_penalty: Scalar[penalty_type],
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

        # skip padding tokens
        if token >= 0:
            var count = rebind[Scalar[DType.int32]](
                compressed_frequency_data[idx[0], 1]
            ).cast[logit_type]()

            var logit = logits[batch_id, token]

            logit -= (
                frequency_penalty.cast[logit_type]() * count
                + presence_penalty.cast[logit_type]()
            )

            logits[batch_id, token] = logit

    var dispatch_shape = IndexList[1](compressed_frequency_data.dim[0]())
    elementwise[
        func=apply_penalties_fn,
        simd_width=1,
        target=target,
        _trace_description="apply_penalties_to_logits",
    ](dispatch_shape, ctx)
